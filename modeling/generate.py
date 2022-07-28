import argparse
import os
import json
import numpy
from models.dalle.text_tokenizer import TextTokenizer
from models.dalle.dalle import MinDalle
import torch
import time
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mega", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--text", type=str, default="Dali painting of WALL·E")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--image-path", type=str, default="images/walle.png")
    parser.add_argument("--root-dir", type=str, default="pretrained")
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()
    print(args)

    # Initialize tokenizer
    root_dir = os.path.join(
        args.root_dir, f"dalle_{'mini' if args.no_mega else 'mega'}"
    )
    with open(os.path.join(root_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)
    with open(os.path.join(root_dir, "merges.txt"), "r") as f:
        merges = f.read().split("\n")[1:-1]
    tokenizer = TextTokenizer(vocab, merges)

    tokens = tokenizer.tokenize(args.text)
    if len(tokens) > 64:
        tokens = tokens[:64]

    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, : len(tokens)] = tokens
    text_tokens = torch.tensor(text_tokens, dtype=torch.long).cuda()

    torch.manual_seed(args.seed)
    model = MinDalle(is_mega=not args.no_mega, root_dir=args.root_dir)

    # Uncomment these lines to convert the model to TorchScript
    # script_model = torch.jit.script(model.eval())
    # torch.jit.save(
    #     script_model,
    #     f"{args.root_dir}/{'dalle_mega' if not args.no_mega else 'dalle_mini'}/scripted.pt",
    # )
    # model = torch.jit.load(
    #     f"{args.root_dir}/{'dalle_mega' if not args.no_mega else 'dalle_mini'}/scripted.pt"
    # )

    start = time.monotonic()
    image = model.forward(text_tokens, args.grid_size, top_k=args.top_k)
    Image.fromarray(image.numpy()).save(args.image_path)
    print(f"Time elapsed: {time.monotonic() - start:.4f}s")
