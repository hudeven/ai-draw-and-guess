import os
from PIL import Image
import numpy
from torch import LongTensor, FloatTensor
import torch
import torch.backends.cudnn, torch.backends.cuda
import json
import argparse
from typing import Iterator
from .text_tokenizer import TextTokenizer
from .bart_encoder import DalleBartEncoder
from .bart_decoder import DalleBartDecoder
from .vqgan_detokenizer import VQGanDetokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True

IMAGE_TOKEN_LENGTH = 256


class MinDalle:
    def __init__(
        self,
        root_dir: str = "../pretrained",
        dtype: torch.dtype = torch.float32,
        is_mega: bool = True,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype

        root_dir = os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}")
        with open(os.path.join(root_dir, "config.json"), "r") as f:
            self.config = argparse.Namespace(**json.load(f))

        # Initialize tokenizer
        with open(os.path.join(root_dir, "vocab.json"), "r") as f:
            vocab = json.load(f)
        with open(os.path.join(root_dir, "merges.txt"), "r") as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)

        # Initialize encoder
        self.encoder = (
            DalleBartEncoder(
                attention_heads=self.config.attention_heads,
                embed_size=self.config.embed_size,
                glu_dim=self.config.glu_dim,
                max_text_length=self.config.max_text_length,
                encoder_vocab_size=self.config.encoder_vocab_size,
                num_layers=self.config.num_layers,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(os.path.join(root_dir, "encoder.pt"))
        self.encoder.load_state_dict(params, strict=False)
        self.encoder = self.encoder.to(device=self.device)
        del params

        # Initialize decoder
        self.decoder = (
            DalleBartDecoder(
                image_vocab_size=self.config.image_vocab_size,
                attention_heads=self.config.attention_heads,
                embed_size=self.config.embed_size,
                glu_dim=self.config.glu_dim,
                num_layers=self.config.num_layers,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(os.path.join(root_dir, "decoder.pt"))
        self.decoder.load_state_dict(params, strict=False)
        self.decoder = self.decoder.to(device=self.device)
        del params

        # Initialize detokenizer
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(os.path.join(root_dir, "detoker.pt"))
        self.detokenizer.load_state_dict(params)
        self.detokenizer = self.detokenizer.to(device=self.device)
        del params

    def generate_image(
        self,
        text: str,
        grid_size: int,
        is_seamless: bool = False,
        temperature: float = 1,
        top_k: int = 256,
        supercondition_factor: int = 16,
    ) -> Iterator[FloatTensor]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.config.max_text_length:
            tokens = tokens[: self.config.max_text_length]
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, : len(tokens)] = tokens
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)
        torch.cuda.empty_cache()

        num_images = grid_size**2
        with torch.cuda.amp.autocast(dtype=self.dtype):
            expanded_indices = [0] * num_images + [1] * num_images
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]
            attention_mask = text_tokens.not_equal(1)
            attention_state = torch.zeros(
                size=(
                    self.config.num_layers,
                    num_images * 4,
                    IMAGE_TOKEN_LENGTH,
                    self.config.embed_size,
                ),
                device=self.device,
            )
            image_tokens = torch.full(
                (IMAGE_TOKEN_LENGTH + 1, num_images),
                self.config.image_vocab_size,
                dtype=torch.long,
                device=self.device,
            )

        token_indices = torch.arange(IMAGE_TOKEN_LENGTH, device=self.device)
        settings = torch.tensor(
            [temperature, top_k, supercondition_factor],
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(IMAGE_TOKEN_LENGTH):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[i + 1], attention_state = self.decoder.forward(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[i],
                    token_index=token_indices[[i]],
                )

        image = self.detokenizer.forward(is_seamless, image_tokens[1:].T)
        return image.to(torch.uint8).to("cpu").numpy()
