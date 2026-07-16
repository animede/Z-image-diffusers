#!/usr/bin/env python3
"""CLI test: Z-Image-Turbo image-to-image (ZImageImg2ImgPipeline).

Usage:
    python test_i2i.py --image ref.png --prompt "..." \\
        [--strength 0.6] [--steps 8] [--guidance 0.0] \\
        [--width 1024] [--height 1024] [--seed -1]
"""
import argparse
import os
import time

import torch
from PIL import Image

from pipeline_manager import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STEPS,
    ZImageConfig,
    get_i2i_pipeline,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEFAULT_PROMPT = "Turn the scene into a vibrant watercolor painting"


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo I2I test")
    parser.add_argument("--image", type=str, required=True, help="Input reference image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=-1, help="-1 = random")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_image = Image.open(args.image).convert("RGB")

    config = ZImageConfig()
    pipe = get_i2i_pipeline(config)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        image=ref_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    )
    image = result.images[0]

    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    out_path = args.output or os.path.join(OUTPUT_DIR, f"i2i_{int(time.time())}.png")
    image.save(out_path)

    print("=" * 60)
    print(f"prompt         : {args.prompt}")
    print(f"input image    : {args.image}")
    print(f"config         : {config}")
    print(f"strength       : {args.strength}")
    print(f"steps          : {args.steps}")
    print(f"guidance_scale : {args.guidance}")
    print(f"size           : {args.width}x{args.height}")
    print(f"elapsed        : {elapsed:.2f}s")
    print(f"peak VRAM      : {peak_mem_gb:.2f} GB")
    print(f"saved to       : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
