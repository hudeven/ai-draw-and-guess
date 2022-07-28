import argparse
from models.dalle.dalle import MinDalle
import torch
import time
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mega", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--text", type=str, default="Dali painting of WALL·E")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--image-path", type=str, default="images/walle.png")
    parser.add_argument("--root-dir", type=str, default="pretrained")
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    model = MinDalle(
        is_mega=not args.no_mega,
        root_dir=args.root_dir,
        dtype=torch.float16 if not args.no_fp16 else torch.float32,
    )
    start = time.monotonic()
    image = model.generate_image(args.text, args.grid_size, top_k=args.top_k)
    Image.fromarray(image).save(args.image_path)
    print(f"Time elapsed: {time.monotonic() - start:.4f}s")