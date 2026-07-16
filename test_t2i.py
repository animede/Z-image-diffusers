#!/usr/bin/env python3
"""CLI test: Z-Image-Turbo text-to-image generation.

Usage:
    python test_t2i.py --prompt "..." [--steps 8] [--guidance 0.0] \\
        [--width 1024] [--height 1024] [--seed -1]
"""
import argparse
import os
import time

import torch

from pipeline_manager import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, ZImageConfig, load_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DEFAULT_PROMPT = (
    "A photorealistic portrait of a red fox wearing a small wizard hat, "
    "standing in a misty autumn forest, cinematic lighting, high detail"
)


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo T2I test")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=-1, help="-1 = random")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = ZImageConfig()
    pipe = load_pipeline(config)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    )
    image = result.images[0]

    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    out_path = args.output or os.path.join(OUTPUT_DIR, f"t2i_{int(time.time())}.png")
    image.save(out_path)

    print("=" * 60)
    print(f"prompt         : {args.prompt}")
    print(f"config         : {config}")
    print(f"steps          : {args.steps}")
    print(f"guidance_scale : {args.guidance}")
    print(f"size           : {args.width}x{args.height}")
    print(f"elapsed        : {elapsed:.2f}s")
    print(f"peak VRAM      : {peak_mem_gb:.2f} GB")
    print(f"saved to       : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
