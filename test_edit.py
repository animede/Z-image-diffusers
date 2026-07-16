#!/usr/bin/env python3
"""CLI test: Z-Image-Turbo mask inpainting ("Edit") via ZImageInpaintPipeline.

No official Z-Image-Edit checkpoint is published as of 2026-07, so "edit" in
this project is implemented as mask inpainting on the Z-Image-Turbo weights
(white mask pixels = region to regenerate). See pipeline_manager.py module
docstring.

If --image / --mask are not given, a synthetic test image with a plain
rectangular mask covering its right half is generated with PIL so this
script can be run standalone without any input assets.

Usage:
    python test_edit.py --image ref.png --mask mask.png --prompt "..." \\
        [--strength 1.0] [--steps 8] [--guidance 0.0]
"""
import argparse
import os
import time

import torch
from PIL import Image, ImageDraw

from pipeline_manager import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STEPS,
    ZImageConfig,
    get_inpaint_pipeline,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEFAULT_PROMPT = "A glowing blue crystal formation"


def make_synthetic_inputs(width=1024, height=1024):
    """Build a plain synthetic base image + a rectangular mask over its right half."""
    image = Image.new("RGB", (width, height), color=(60, 90, 60))
    draw = ImageDraw.Draw(image)
    draw.ellipse((width // 8, height // 8, width // 2, height // 2), fill=(180, 140, 60))

    mask = Image.new("L", (width, height), color=0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rectangle((width // 2, 0, width, height), fill=255)  # white = region to edit
    return image, mask


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Edit (inpaint) test")
    parser.add_argument("--image", type=str, default=None, help="Input image path (default: synthetic)")
    parser.add_argument("--mask", type=str, default=None, help="Mask image path, white=edit region (default: synthetic)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=-1, help="-1 = random")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.image and args.mask:
        base_image = Image.open(args.image).convert("RGB")
        mask_image = Image.open(args.mask).convert("L")
    else:
        print("[test_edit] --image/--mask not given; generating a synthetic base image + rectangular mask")
        base_image, mask_image = make_synthetic_inputs(args.width, args.height)
        base_image.save(os.path.join(OUTPUT_DIR, "edit_synthetic_base.png"))
        mask_image.save(os.path.join(OUTPUT_DIR, "edit_synthetic_mask.png"))

    config = ZImageConfig()
    pipe = get_inpaint_pipeline(config)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        image=base_image,
        mask_image=mask_image,
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

    out_path = args.output or os.path.join(OUTPUT_DIR, f"edit_{int(time.time())}.png")
    image.save(out_path)

    print("=" * 60)
    print(f"prompt         : {args.prompt}")
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
