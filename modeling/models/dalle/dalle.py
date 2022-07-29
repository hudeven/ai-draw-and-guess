import os
import numpy
import torch
import torch.nn as nn
import torch.backends.cudnn, torch.backends.cuda
import json
from .bart_encoder import DalleBartEncoder
from .bart_decoder import DalleBartDecoder
from .vqgan_detokenizer import VQGanDetokenizer
from models.dalle.text_tokenizer import TextTokenizer


torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True


class MinDalle(nn.Module):
    def __init__(self, root_dir="../pretrained", is_mega=True, is_reusable: bool = True):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_reusable = is_reusable

        root_dir = os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}")
        with open(os.path.join(root_dir, "config.json"), "r") as f:
            self.config = json.load(f)

        # Initialize encoder
        self.encoder = DalleBartEncoder(
            attention_heads=self.config["attention_heads"],
            embed_size=self.config["embed_size"],
            glu_dim=self.config["glu_dim"],
            max_text_length=self.config["max_text_length"],
            encoder_vocab_size=self.config["encoder_vocab_size"],
            num_layers=self.config["num_layers"],
            device=self.device,
        )
        params = torch.load(os.path.join(root_dir, "encoder.pt"))
        self.encoder.load_state_dict(params, strict=False)
        self.encoder = self.encoder.to(torch.float16).to(device=self.device).eval()

        # Initialize decoder
        self.decoder = DalleBartDecoder(
            image_vocab_size=self.config["image_vocab_size"],
            attention_heads=self.config["attention_heads"],
            embed_size=self.config["embed_size"],
            glu_dim=self.config["glu_dim"],
            num_layers=self.config["num_layers"],
            device=self.device,
        )
        params = torch.load(os.path.join(root_dir, "decoder.pt"))
        self.decoder.load_state_dict(params, strict=False)
        self.decoder = self.decoder.to(torch.float16).to(device=self.device).eval()

        # Initialize detokenizer
        self.detokenizer = VQGanDetokenizer()
        params = torch.load(os.path.join(root_dir, "detoker.pt"))
        self.detokenizer.load_state_dict(params)
        self.detokenizer = self.detokenizer.to(device=self.device).eval()

    def forward(
        self,
        text_tokens,
        grid_size: int,
        temperature: float = 1,
        top_k: int = 256,
        supercondition_factor: int = 16,
        progressive_outputs: bool = False,
        is_seamless: bool = False,
        is_verbose: bool = True,
    ):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            encoder_state = self.encoder.forward(text_tokens)

        encoder_state = encoder_state.to(torch.float16)
        num_images = int(grid_size * grid_size)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            expanded_indices = [0] * num_images + [1] * num_images
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]
            attention_mask = text_tokens.not_equal(1)
            attention_state = encoder_state.new_zeros(
                size=(
                    self.config["num_layers"],
                    int(num_images * 4),
                    256,
                    self.config["embed_size"],
                )
            )
            image_tokens = torch.full(
                (257, int(num_images)),
                int(self.config["image_vocab_size"]),
                dtype=torch.long,
                device=self.device,
            )

        token_indices = torch.arange(256, device=self.device)
        settings = torch.tensor(
            [temperature, float(top_k), float(supercondition_factor)],
            dtype=torch.float16,
            device=self.device,
        )
        for i in range(256):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_tokens[i + 1], attention_state = self.decoder.forward(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[i],
                    token_index=token_indices[[i]],
                )

            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((i + 1) % 32 == 0 and progressive_outputs) or i + 1 == 256:
                    yield self.image_grid_from_tokens(
                        image_tokens=image_tokens[1:].T,
                        is_seamless=is_seamless,
                        is_verbose=is_verbose
                    )
        # image = self.detokenizer.forward(image_tokens[1:].T)
        # image = image.to(torch.uint8).to("cpu")
        # return image


    def image_grid_from_tokens(
        self,
        image_tokens,
        is_seamless: bool,
        is_verbose: bool = False,
    ):
        if not self.is_reusable: del self.decoder
        torch.cuda.empty_cache()
        if not self.is_reusable: self.init_detokenizer()
        if is_verbose: print("detokenizing image")
        images = self.detokenizer.forward(image_tokens)
        images = images.to(torch.uint8).to("cpu")
        if not self.is_reusable: del self.detokenizer
        return images


def get_tokenizer(root_dir):
    with open(os.path.join(root_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)
    with open(os.path.join(root_dir, "merges.txt"), "r") as f:
        merges = f.read().split("\n")[1:-1]
    return TextTokenizer(vocab, merges)


def prepare_tokens(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > 64:
        tokens = tokens[:64]

    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, : len(tokens)] = tokens
    text_tokens = torch.tensor(text_tokens, dtype=torch.long).cuda()
    return text_tokens
