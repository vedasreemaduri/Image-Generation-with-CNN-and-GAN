"""Colorize a single RGB image using a trained generator checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train import Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/generator.pt"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/colorized.png"))
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("image-size must be >=32 and divisible by 32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["generator"])
    model.eval()
    image = Image.open(args.input).convert("RGB")
    original_size = image.size
    lab = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC).convert("LAB")
    values = np.asarray(lab, dtype=np.float32)
    light = torch.from_numpy(values[..., 0] / 127.5 - 1).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        chroma = model(light)[0].cpu().numpy().transpose(1, 2, 0)
    ab = ((chroma * 127.5) + 128).clip(0, 255).astype(np.uint8)
    L = ((values[..., 0]).clip(0, 255)).astype(np.uint8)
    result = Image.fromarray(np.dstack((L, ab)), mode="LAB").convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.resize(original_size, Image.Resampling.LANCZOS).save(args.output)
    print(f"Saved colorized image to {args.output}")

if __name__ == "__main__":
    main()
