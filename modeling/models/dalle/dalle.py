import io
import os
import numpy
import torch
import torch.nn as nn
import torch.backends.cudnn, torch.backends.cuda
import json

from PIL import Image
from typing import List
from torch import Tensor
from torch.profiler import record_function
from .bart_encoder import DalleBartEncoder
from .bart_decoder import DalleBartDecoder
from .vqgan_detokenizer import VQGanDetokenizer
from .text_tokenizer import TextTokenizer


torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True


class MinDalle(nn.Module):
    def __init__(self, root_dir="../pretrained", is_mega=True, is_reusable: bool = True, device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_reusable = is_reusable

        root_dir = os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}")
        self.root_dir = root_dir
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

        if is_reusable:
            self.init_detokenizer()

    def init_detokenizer(self):
        self.detokenizer = VQGanDetokenizer()
        params = torch.load(os.path.join(self.root_dir, "detoker.pt"))
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
    ) -> List[Tensor]:
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
        
        with record_function("decoding"):
            if progressive_outputs:
                output = self.get_progressive_output(
                    image_tokens,
                    settings,
                    attention_mask,
                    encoder_state,
                    attention_state,
                    token_indices,
                )
            else:
                output = self.get_final_output(
                    image_tokens,
                    settings,
                    attention_mask,
                    encoder_state,
                    attention_state,
                    token_indices,
                )
        return output

    
    def get_final_output(self, image_tokens, settings, attention_mask, encoder_state, attention_state, token_indices) -> List[Tensor]:
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

        image = self.detokenizer.forward(image_tokens[1:].T)
        image = image.to(torch.uint8).to("cpu")
        return [image]

    
    @torch.jit.unused
    def get_progressive_output(self, image_tokens, settings, attention_mask, encoder_state, attention_state, token_indices) -> List[Tensor]:
        for i in range(256):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_tokens[i + 1], attention_state = self.decoder.forward(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[i],
                    token_index=token_indices[[i]],
                )

            if ((i + 1) % 32 == 0) or i + 1 == 256:
                print(f"yield progressive output: {i}")
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    yield self.image_grid_from_tokens(
                        image_tokens=image_tokens[1:].T,
                        is_seamless=False,
                        is_verbose=True,
                    )

                    
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


def prepare_tokens(tokenizer, text, device=None):
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokenizer.tokenize(text)
    if len(tokens) > 64:
        tokens = tokens[:64]

    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, : len(tokens)] = tokens
    text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
    return text_tokens


def post_process(img_tensors):
    for t in img_tensors:
        img = Image.fromarray(t.numpy())
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        yield buffer.getvalue()
