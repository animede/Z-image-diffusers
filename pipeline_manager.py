"""
Z-Image-Turbo diffusers pipeline manager.

`Tongyi-MAI/Z-Image-Turbo` is a distilled (turbo) text-to-image model: it is
designed to run in as few as 8 denoising steps with `guidance_scale=0.0`
(classifier-free guidance disabled). Setting `guidance_scale > 1` re-enables
`negative_prompt` support but requires two transformer forward passes per
step instead of one, roughly doubling generation time -- turbo models are
tuned to look good without it, so the default in this project is
`guidance_scale=0.0`.

Three tasks are supported, all backed by pipeline classes that share the
exact same component set (scheduler, vae, text_encoder, tokenizer,
transformer):
  - T2I     : ZImagePipeline
  - I2I     : ZImageImg2ImgPipeline
  - Edit    : ZImageInpaintPipeline (mask inpainting -- see note below)

Only the base ZImagePipeline is loaded from the Hub. The I2I and Edit
pipelines are constructed lazily from `base_pipe.components` (a dict of
references, not copies), so VRAM usage does not multiply per task -- all
three wrapper objects share the same underlying nn.Module instances. Any
offload hooks / attention backend / torch.compile applied to the base
pipeline before deriving the others are therefore already in effect for
I2I and Edit too (the hooks are attached to the shared modules, not to the
pipeline wrapper).

Note on "Edit": as of 2026-07 Tongyi-MAI has not published a dedicated
Z-Image-Edit checkpoint, so "edit" in this project is implemented as mask
inpainting via ZImageInpaintPipeline (white mask pixels = region to
regenerate) using the same Z-Image-Turbo weights. If/when an official Edit
checkpoint ships, add a new entry to `PRECISION_BUILDERS`-style registry or
a parallel MODEL_ID_EDIT and swap `_get_inpaint_pipeline`'s source pipeline
without touching call sites.

Design notes (mirrors ~/flux2_diffusers/pipeline_manager.py):
  - `PRECISION_BUILDERS` is a registry keyed by precision name. Add a new
    entry (e.g. "fp8") without touching call sites.
  - `_apply_offload` / `_apply_attention_backend` / `_apply_compile` /
    `_apply_vae_tiling` are separate hooks so each optimization can be
    toggled independently and combined freely via ZImageConfig.
  - The base pipeline is a process-wide singleton; I2I / Edit pipelines are
    separate singletons derived from it on first use (lazy construction).
"""
import os
import threading
import time
import traceback

import torch
from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# Turbo defaults: distilled for few-step, CFG-free sampling.
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0

# Attention backends verified to work on this machine (RTX PRO 6000
# Blackwell, sm_120, torch 2.9 + diffusers 0.39). Same findings as
# ~/flux2_diffusers/pipeline_manager.py (identical GPU/driver):
#   default        : leave diffusers' dispatch untouched (native SDPA)
#   native         : torch SDPA, backend auto-selected by torch
#   _native_flash  : force SDPA flash-attention kernel
#   _native_cudnn  : force SDPA cuDNN attention kernel
#   _native_efficient : force SDPA memory-efficient kernel
#   flex           : torch FlexAttention
#   xformers       : xformers memory_efficient_attention
# NOT working on this machine (fall back to SDPA with a warning):
#   sage / sage_varlen / _sage_* : installed sageattention build has no
#       sm_120 kernels ("no kernel image is available"), and the failure is
#       an async CUDA launch error that cannot be caught reliably -- reject
#       up front instead of crashing the first generation.
#   flash / flash_varlen : flash-attn package not installed
#   *_hub (flash_hub, sage_hub, _flash_3_hub, ...) : require the `kernels`
#       package; _flash_3_hub additionally targets Hopper (sm_90), not
#       Blackwell
KNOWN_GOOD_ATTN_BACKENDS = (
    "default",
    "native",
    "_native_flash",
    "_native_cudnn",
    "_native_efficient",
    "flex",
    "xformers",
)

KNOWN_BAD_ATTN_BACKEND_PREFIXES = ("sage", "_sage")

_lock = threading.Lock()
_base_pipeline = None       # ZImagePipeline (T2I)
_i2i_pipeline = None        # ZImageImg2ImgPipeline, derived from _base_pipeline
_inpaint_pipeline = None    # ZImageInpaintPipeline, derived from _base_pipeline
_pipeline_config_repr = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class ZImageConfig:
    """Runtime configuration, overridable via environment variables.

    ZIMAGE_PRECISION  : "bf16" (default; full precision, ~20GB, fastest on a
                         96GB card) | "bnb-4bit" (bitsandbytes 4bit
                         quantization of the transformer + text encoder at
                         load time; lower VRAM, some quality/speed tradeoff)
    ZIMAGE_OFFLOAD    : "none" (default; everything resident on GPU --
                         fastest, ~21GB, recommended for this 96GB machine) |
                         "model" (alias "model_cpu_offload"; one component on
                         GPU at a time) | "group" (alias "group_offload";
                         leaf_level + CUDA-stream prefetch, lowest VRAM)
    ZIMAGE_ATTN       : attention backend for the transformer. "default"
                        (= native SDPA), "native", "_native_flash",
                        "_native_cudnn", "_native_efficient", "flex",
                        "xformers", ... (any diffusers AttentionBackendName).
                        Unavailable backends warn and fall back to SDPA.
    ZIMAGE_COMPILE    : "0" (default) | "1" to torch.compile the transformer
                        (regional compilation of repeated blocks via
                        `compile_repeated_blocks`, falling back to plain
                        torch.compile on failure).
    ZIMAGE_VAE_TILING : "0" (default) | "1" to enable_slicing() +
                        enable_tiling() on the VAE (useful for
                        higher-than-1024 resolutions / lower VRAM decode).
    ZIMAGE_DEVICE     : "cuda" (default)
    """

    def __init__(self):
        self.precision = _env("ZIMAGE_PRECISION", "bf16")
        self.offload = _env("ZIMAGE_OFFLOAD", "none")
        self.attention_backend = _env("ZIMAGE_ATTN", "default")
        self.compile = _env("ZIMAGE_COMPILE", "0") == "1"
        self.vae_tiling = _env("ZIMAGE_VAE_TILING", "0") == "1"
        self.device = _env("ZIMAGE_DEVICE", "cuda")

    def __repr__(self):
        return (
            f"ZImageConfig(precision={self.precision!r}, offload={self.offload!r}, "
            f"attention_backend={self.attention_backend!r}, compile={self.compile}, "
            f"vae_tiling={self.vae_tiling}, device={self.device!r})"
        )

    def as_dict(self):
        return {
            "precision": self.precision,
            "offload": self.offload,
            "attention_backend": self.attention_backend,
            "compile": self.compile,
            "vae_tiling": self.vae_tiling,
            "device": self.device,
        }


def _build_bf16(config: "ZImageConfig") -> ZImagePipeline:
    return ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)


def _build_bnb_4bit(config: "ZImageConfig") -> ZImagePipeline:
    # Quantize the transformer and text encoder to 4bit at load time via
    # diffusers' pipeline-level quantization config; torch_dtype controls the
    # compute dtype for non-quantized modules (VAE, etc.) and the compute
    # dtype used internally by the 4bit layers.
    from diffusers.quantizers import PipelineQuantizationConfig

    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs=dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        components_to_quantize=["transformer", "text_encoder"],
    )
    return ZImagePipeline.from_pretrained(
        MODEL_ID, quantization_config=quant_config, torch_dtype=torch.bfloat16
    )


# Registry so new precisions (e.g. fp8) can be added without touching call
# sites (load_pipeline / test scripts / api_server / ui).
PRECISION_BUILDERS = {
    "bf16": _build_bf16,
    "bnb-4bit": _build_bnb_4bit,
}


def _apply_offload(pipe: ZImagePipeline, config: "ZImageConfig") -> None:
    """Place the pipeline on the device according to the offload mode.

    - "none" (default): everything resident on GPU. Fastest, ~21GB VRAM for
      bf16 -- the recommended setting on a 96GB card.
    - "model" (alias "model_cpu_offload"): one whole component on GPU at a
      time via `enable_model_cpu_offload()`. Balanced speed/VRAM.
    - "group" (alias "group_offload"): stream groups of layers on/off the
      GPU (`leaf_level` + CUDA-stream prefetch). Lowest VRAM, slowest --
      reference configuration for low-VRAM machines.
    """
    mode = (config.offload or "none").lower()
    if mode == "none":
        pipe.to(config.device)
    elif mode in ("model", "model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif mode in ("group", "group_offload"):
        from diffusers.hooks import apply_group_offloading

        # ZIMAGE_GROUP_STREAM=1 (default): CUDA-stream prefetch, pins
        # offloaded weights in CPU RAM (fine here: Z-Image-Turbo bf16 is
        # ~20GB total, well under the 62GB system RAM).
        use_stream = _env("ZIMAGE_GROUP_STREAM", "1") == "1"
        onload_device = torch.device(config.device)
        offload_device = torch.device("cpu")
        common = dict(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=use_stream,
        )
        # transformer is a diffusers ModelMixin -> native method
        pipe.transformer.enable_group_offload(**common)
        # text_encoder (a transformers PreTrainedModel) -> functional API
        apply_group_offloading(pipe.text_encoder, **common)
        # VAE is small; keep it resident
        pipe.vae.to(onload_device)
        print(f"[pipeline_manager] group offload enabled (leaf_level, use_stream={use_stream})")
    else:
        raise ValueError(f"Unknown offload mode: {config.offload!r}")


def _smoke_test_attention_backend(backend: str, device: str) -> None:
    """Run one tiny attention call through the requested backend.

    Backends can pass diffusers' requirement check (package installed) but
    still fail at kernel launch on unsupported GPUs -- e.g. a sageattention
    build without sm_120 kernels raises "no kernel image is available" only
    when the kernel is first launched. Launch failures of this kind are
    non-sticky, so catching them here is safe and lets us fall back cleanly
    instead of crashing the first generation.

    Shape approximates a real Z-Image-Turbo call: ZImageTransformer2DModel
    uses n_heads=30, dim=3840 (head_dim=128); seq_len=4096 is in the range
    of a 1024x1024 generation's patchified latent + text tokens.
    """
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        dispatch_attention_fn,
    )

    q = torch.randn(1, 4096, 30, 128, dtype=torch.bfloat16, device=device)
    out = dispatch_attention_fn(q, q, q, backend=AttentionBackendName(backend))
    torch.cuda.synchronize()
    del q, out


def _apply_attention_backend(pipe: ZImagePipeline, config: "ZImageConfig") -> None:
    """Set the transformer attention backend.

    Unknown names / missing dependencies are rejected by diffusers'
    `set_attention_backend`; kernel-launch failures on this GPU are caught by
    a tiny smoke test. Either way the failure logs a warning and leaves the
    default SDPA dispatch in place instead of raising.
    """
    backend = (config.attention_backend or "default").lower()
    if backend in ("default", ""):
        return

    if backend.startswith(KNOWN_BAD_ATTN_BACKEND_PREFIXES):
        print(
            f"[pipeline_manager] warning: attention backend {backend!r} is known "
            f"to be broken on this GPU (sm_120 kernels missing in the installed "
            f"build; async CUDA launch failure); falling back to default SDPA."
        )
        return

    if backend not in KNOWN_GOOD_ATTN_BACKENDS:
        print(
            f"[pipeline_manager] note: attention backend {backend!r} is not in "
            f"the verified-working list {list(KNOWN_GOOD_ATTN_BACKENDS)} for this "
            f"machine; trying anyway."
        )
    try:
        pipe.transformer.set_attention_backend(backend)
        _smoke_test_attention_backend(backend, config.device)
        print(f"[pipeline_manager] attention backend set to {backend!r}")
    except Exception as e:
        print(
            f"[pipeline_manager] warning: attention backend {backend!r} is not "
            f"usable ({type(e).__name__}: {e}); falling back to default SDPA."
        )
        try:
            pipe.transformer.reset_attention_backend()
        except Exception:
            pass


def _apply_compile(pipe: ZImagePipeline, config: "ZImageConfig") -> None:
    """torch.compile the transformer.

    Prefers diffusers' regional compilation (`compile_repeated_blocks`),
    which compiles `ZImageTransformerBlock` (the repeated block class of
    ZImageTransformer2DModel) once and reuses the compiled graph, giving
    most of the speedup at a fraction of the compile time. Falls back to
    `transformer.compile()` if regional compilation fails.
    """
    if not config.compile:
        return

    try:
        pipe.transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
        print("[pipeline_manager] transformer compiled (regional, compile_repeated_blocks)")
        return
    except Exception as e:
        print(
            f"[pipeline_manager] regional compilation failed "
            f"({type(e).__name__}: {e}); falling back to plain torch.compile"
        )
        traceback.print_exc()

    try:
        pipe.transformer.compile()
        print("[pipeline_manager] transformer compiled (plain torch.compile)")
    except Exception as e:
        print(
            f"[pipeline_manager] warning: torch.compile failed "
            f"({type(e).__name__}: {e}); continuing without compilation."
        )


def _apply_vae_tiling(pipe: ZImagePipeline, config: "ZImageConfig") -> None:
    """Enable VAE slicing + tiling for high-resolution / low-VRAM decode."""
    if not config.vae_tiling:
        return
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("[pipeline_manager] VAE slicing + tiling enabled")
    except Exception as e:
        print(
            f"[pipeline_manager] warning: could not enable VAE tiling "
            f"({type(e).__name__}: {e}); continuing without it."
        )


def load_pipeline(config: "ZImageConfig" = None, force_reload: bool = False) -> ZImagePipeline:
    """Load (or return the cached) ZImagePipeline singleton for the process.

    This is the base T2I pipeline. Call `get_i2i_pipeline()` /
    `get_inpaint_pipeline()` to obtain the derived I2I / Edit(inpaint)
    pipelines, which share this pipeline's components.
    """
    global _base_pipeline, _pipeline_config_repr

    config = config or ZImageConfig()

    with _lock:
        if _base_pipeline is not None and not force_reload:
            if _pipeline_config_repr != repr(config):
                print(
                    f"[pipeline_manager] warning: pipeline already loaded with "
                    f"config={_pipeline_config_repr!r}; requested config "
                    f"{config!r} was ignored. Call "
                    f"load_pipeline(config, force_reload=True) to reload."
                )
            return _base_pipeline

        if config.precision not in PRECISION_BUILDERS:
            raise ValueError(
                f"Unknown precision {config.precision!r}. "
                f"Available: {list(PRECISION_BUILDERS)}"
            )

        print(f"[pipeline_manager] loading ZImagePipeline: {config}")
        t0 = time.time()

        builder = PRECISION_BUILDERS[config.precision]
        pipe = builder(config)

        _apply_offload(pipe, config)
        _apply_attention_backend(pipe, config)
        _apply_compile(pipe, config)
        _apply_vae_tiling(pipe, config)

        _base_pipeline = pipe
        _pipeline_config_repr = repr(config)
        print(f"[pipeline_manager] loaded in {time.time() - t0:.1f}s")

    return _base_pipeline


def get_i2i_pipeline(config: "ZImageConfig" = None) -> ZImageImg2ImgPipeline:
    """Return the ZImageImg2ImgPipeline singleton, derived from the base pipeline.

    Constructed lazily from `base_pipe.components` (shared module references,
    not copies) so VRAM usage does not multiply; any offload hooks /
    attention backend / compile already applied to the base pipeline are in
    effect here too since the underlying nn.Modules are the same objects.
    """
    global _i2i_pipeline

    base = load_pipeline(config)
    with _lock:
        if _i2i_pipeline is None:
            _i2i_pipeline = ZImageImg2ImgPipeline(**base.components)
            print("[pipeline_manager] ZImageImg2ImgPipeline derived from base pipeline components")
    return _i2i_pipeline


def get_inpaint_pipeline(config: "ZImageConfig" = None) -> ZImageInpaintPipeline:
    """Return the ZImageInpaintPipeline singleton, derived from the base pipeline.

    Used for the "Edit" feature (mask inpainting) since no official
    Z-Image-Edit checkpoint is published yet -- see module docstring.
    """
    global _inpaint_pipeline

    base = load_pipeline(config)
    with _lock:
        if _inpaint_pipeline is None:
            _inpaint_pipeline = ZImageInpaintPipeline(**base.components)
            print("[pipeline_manager] ZImageInpaintPipeline derived from base pipeline components")
    return _inpaint_pipeline


def get_loaded_config_repr():
    """Return the repr() of the ZImageConfig the currently loaded pipeline used, or None."""
    return _pipeline_config_repr


def is_loaded() -> bool:
    return _base_pipeline is not None
