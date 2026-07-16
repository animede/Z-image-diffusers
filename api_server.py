#!/usr/bin/env python3
"""Z-Image-Turbo FastAPI server.

Thin HTTP wrapper around pipeline_manager.py. The model is lazily loaded on
the first request (startup itself is cheap). All GPU calls are serialized
behind a single `threading.Lock` -- Z-Image-Turbo generation is fast (a few
seconds at 8 steps) but the GPU can only run one generation at a time
regardless, so requests queue rather than fight over VRAM/streams.

Endpoints:
  GET  /health    -> {"status": "ok", "model_loaded": bool, "config": {...}}
  POST /api/t2i   -> text-to-image (ZImagePipeline)
  POST /api/i2i   -> image-to-image (ZImageImg2ImgPipeline)
  POST /api/edit  -> mask inpainting (ZImageInpaintPipeline) -- see
                     pipeline_manager.py docstring for why "edit" == inpaint

Run:
    venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import os
import threading
import time
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from pipeline_manager import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STEPS,
    ZImageConfig,
    get_i2i_pipeline,
    get_inpaint_pipeline,
    is_loaded,
    load_pipeline,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Z-Image-Turbo API", version="1.0")

# Serializes all GPU calls: only one generation runs at a time regardless of
# how many requests arrive concurrently.
_gpu_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = DEFAULT_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    width: int = 1024
    height: int = 1024
    seed: int = -1
    num_images: int = 1


class I2IRequest(T2IRequest):
    image: str  # base64 PNG/JPEG
    strength: float = 0.6


class EditRequest(T2IRequest):
    image: str  # base64 PNG/JPEG
    mask: str  # base64, white = region to edit
    strength: float = 1.0


class GenerationResponse(BaseModel):
    images: List[str]
    seed: int
    elapsed_sec: float
    peak_vram_gb: float
    saved_paths: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_b64_image(data: str, mode: str = "RGB") -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert(mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {type(e).__name__}: {e}")


def _encode_pil_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_generator(seed: int):
    if seed is None or int(seed) < 0:
        actual_seed = torch.seed() % (2 ** 31 - 1)
    else:
        actual_seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
    return generator, actual_seed


def _save_images(images: List[Image.Image], prefix: str) -> List[str]:
    ts = int(time.time() * 1000)
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}_{i}.png")
        img.save(path)
        paths.append(path)
    return paths


def _validate_prompt(prompt: str):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")


def _validate_dims(width: int, height: int):
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width/height must be between 256 and 2048")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    config = ZImageConfig()
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "config": config.as_dict(),
    }


@app.post("/api/t2i", response_model=GenerationResponse)
def api_t2i(req: T2IRequest):
    _validate_prompt(req.prompt)
    _validate_dims(req.width, req.height)
    if not (1 <= req.num_images <= 8):
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 8")

    try:
        with _gpu_lock:
            pipe = load_pipeline(ZImageConfig())
            generator, used_seed = _make_generator(req.seed)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                num_images_per_prompt=req.num_images,
                generator=generator,
            )
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        saved_paths = _save_images(result.images, "t2i")
        return GenerationResponse(
            images=[_encode_pil_image(img) for img in result.images],
            seed=used_seed,
            elapsed_sec=elapsed,
            peak_vram_gb=peak_vram_gb,
            saved_paths=saved_paths,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"T2I generation failed: {type(e).__name__}: {e}")


@app.post("/api/i2i", response_model=GenerationResponse)
def api_i2i(req: I2IRequest):
    _validate_prompt(req.prompt)
    _validate_dims(req.width, req.height)
    if not (0.0 <= req.strength <= 1.0):
        raise HTTPException(status_code=400, detail="strength must be between 0 and 1")

    input_image = _decode_b64_image(req.image)

    try:
        with _gpu_lock:
            pipe = get_i2i_pipeline(ZImageConfig())
            generator, used_seed = _make_generator(req.seed)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                image=input_image,
                strength=req.strength,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                num_images_per_prompt=req.num_images,
                generator=generator,
            )
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        saved_paths = _save_images(result.images, "i2i")
        return GenerationResponse(
            images=[_encode_pil_image(img) for img in result.images],
            seed=used_seed,
            elapsed_sec=elapsed,
            peak_vram_gb=peak_vram_gb,
            saved_paths=saved_paths,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"I2I generation failed: {type(e).__name__}: {e}")


@app.post("/api/edit", response_model=GenerationResponse)
def api_edit(req: EditRequest):
    _validate_prompt(req.prompt)
    _validate_dims(req.width, req.height)
    if not (0.0 <= req.strength <= 1.0):
        raise HTTPException(status_code=400, detail="strength must be between 0 and 1")

    input_image = _decode_b64_image(req.image, mode="RGB")
    mask_image = _decode_b64_image(req.mask, mode="L")

    try:
        with _gpu_lock:
            pipe = get_inpaint_pipeline(ZImageConfig())
            generator, used_seed = _make_generator(req.seed)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                image=input_image,
                mask_image=mask_image,
                strength=req.strength,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                num_images_per_prompt=req.num_images,
                generator=generator,
            )
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        saved_paths = _save_images(result.images, "edit")
        return GenerationResponse(
            images=[_encode_pil_image(img) for img in result.images],
            seed=used_seed,
            elapsed_sec=elapsed,
            peak_vram_gb=peak_vram_gb,
            saved_paths=saved_paths,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit generation failed: {type(e).__name__}: {e}")
