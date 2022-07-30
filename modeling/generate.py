import argparse
import os
import json
import numpy
from models.dalle.dalle import MinDalle, get_tokenizer, prepare_tokens
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
    parser.add_argument("--image-path", type=str, default="images/walle")
    parser.add_argument("--root-dir", type=str, default="pretrained")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--ts", action="store_true", help="whether to torchscript the model")
    args = parser.parse_args()
    print(args)

    # Initialize tokenizer
    root_dir = os.path.join(
        args.root_dir, f"dalle_{'mini' if args.no_mega else 'mega'}"
    )

    tokenizer = get_tokenizer(root_dir)
    tokens = prepare_tokens(tokenizer, args.text)

    torch.manual_seed(args.seed)
    model = MinDalle(is_mega=not args.no_mega, root_dir=args.root_dir)
    model.eval()

    # Uncomment these lines to convert the model to TorchScript
    if args.ts:
        model = torch.jit.script(model)
        torch.jit.save(
            model,
            f"{args.root_dir}/{'dalle_mega' if not args.no_mega else 'dalle_mini'}/scripted.pt",
        )
        model = torch.jit.load(
            f"{args.root_dir}/{'dalle_mega' if not args.no_mega else 'dalle_mini'}/scripted.pt"
        )
        model.eval()

    start = time.monotonic()
    for i, image in enumerate(model(tokens, args.grid_size, top_k=args.top_k, progressive_outputs=not args.ts)):
         Image.fromarray(image.numpy()).save(args.image_path + "_" + str(i) + ".png") 

    print(f"Time elapsed: {time.monotonic() - start:.4f}s")
