"""
generate.py — Stable Diffusion style illustration generator (Mode 2).

Uses ``runwayml/stable-diffusion-v1-5`` via the ``diffusers`` library.
Device priority: mps (Apple Silicon) → cpu (fallback with a console warning).

MacBook Air constraints enforced here
──────────────────────────────────────
  • Pipeline loaded LAZILY on first request, NOT at server startup
    (4 GB download + model load takes ~30 s — too slow for boot).
  • num_inference_steps hard-capped at MAX_STEPS = 25.
  • guidance_scale fixed at GUIDANCE = 7.5.
  • attention_slicing + vae_slicing enabled to cut peak RAM ~40 %.
  • torch.mps.empty_cache() called in a ``finally`` block after every run.
  • ``unload_sd_pipeline()`` drops the cached pipeline so other models
    (CLIP / YOLO) can load without competing for RAM.
  • NSFW safety checker disabled — saves ~300 MB RAM.

Generation is slow (20–40 s on MacBook Air), so all public async entry-points
delegate to ``run_in_executor`` so the FastAPI event loop stays free.
The ``/styling/illustrate`` route uses a BackgroundTask + status dict so the
frontend can poll without blocking.
"""

import asyncio
import gc
import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_STEPS   = 25
GUIDANCE    = 7.5
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

POSITIVE_SUFFIX = (
    "fashion editorial photograph, flat lay styling, studio lighting, "
    "white background, high detail, sharp focus"
)
NEGATIVE_PROMPT = (
    "text, watermark, logo, blurry, low quality, extra fingers, "
    "deformed, ugly, duplicate, extra limbs, mutation, bad anatomy"
)

_ROOT      = Path(__file__).parents[2]
OUTPUT_DIR = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory generation job registry ────────────────────────────────────────
# { generation_id: { "status": queued|running|done|failed, "path": str|None, "error": str|None } }
_gen_jobs: dict[str, dict] = {}
MAX_GEN_JOBS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    """Return ``'mps'`` on Apple Silicon Metal, ``'cpu'`` otherwise."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    logger.warning(
        "MPS not available — SD will run on CPU (20–60 s per image). "
        "Install PyTorch with MPS support for Apple Silicon speed."
    )
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader  (cached — loaded once, unloaded on demand)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sd_pipeline():
    """
    Load Stable Diffusion v1-5 and cache it in memory.

    Called lazily on first generation request.  Uses ``lru_cache(maxsize=1)``
    so ``cache_clear()`` in ``unload_sd_pipeline()`` drops the only reference
    and Python GC can reclaim the ~2–4 GB.

    Returns:
        A ``StableDiffusionPipeline`` moved to the correct device.

    Raises:
        RuntimeError: If ``diffusers`` / ``torch`` are not installed.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers and torch are required for AI illustration.\n"
            "Run:  pip install diffusers transformers accelerate torch"
        ) from exc

    device = _get_device()
    # float16 on MPS shaves ~50 % RAM vs float32 with negligible quality loss
    dtype  = torch.float16 if device == "mps" else torch.float32

    logger.info(
        "Loading Stable Diffusion v1-5 on %s (dtype=%s) — first load ~30 s…",
        device, dtype,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,           # disable NSFW checker → saves ~300 MB RAM
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Memory optimisations — safe on all diffusers versions ≥ 0.12
    pipe.enable_attention_slicing()    # ~40 % peak RAM reduction
    try:
        pipe.enable_vae_slicing()      # further reduces VAE decode peak
    except AttributeError:
        pass                           # older diffusers may not have this

    logger.info("Stable Diffusion v1-5 ready on %s", device)
    return pipe


def is_sd_loaded() -> bool:
    """Return True if the SD pipeline is currently in memory."""
    # lru_cache stores in __wrapped__'s cache — check via cache_info
    return _load_sd_pipeline.cache_info().currsize > 0


def unload_sd_pipeline() -> None:
    """
    Drop the SD pipeline from memory.

    Call this before loading CLIP / YOLO so they don't compete for RAM.
    Clears lru_cache → triggers GC → empties MPS cache.
    """
    _load_sd_pipeline.cache_clear()
    try:
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    logger.info("SD pipeline unloaded from memory")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sd_prompt(
    outfit_json: dict,
    item_metadata: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """
    Construct (positive_prompt, negative_prompt) from an outfit JSON dict.

    Positive prompt components (in order):
      1. vibe_tags joined with commas
      2. ``{color} {category}`` description of each item (from metadata)
      3. styling_techniques joined
      4. confidence mood word
      5. POSITIVE_SUFFIX  (editorial / lighting / focus keywords)

    Args:
        outfit_json:   Structured outfit dict returned by Ollama.
        item_metadata: Optional list of ChromaDB metadata dicts (parallel to
                       ``outfit_json["outfit"]``).  Accepts both flat and
                       nested ``{"metadata": {...}}`` shapes.

    Returns:
        ``(positive_prompt, negative_prompt)`` tuple of strings.
    """
    parts: list[str] = []

    # 1. Vibe tags
    vibes = outfit_json.get("vibe_tags") or []
    if vibes:
        parts.append(", ".join(str(v) for v in vibes))

    # 2. Item colour + category descriptions
    if item_metadata:
        descriptions: list[str] = []
        for raw in item_metadata:
            # Normalise nested shape
            m = raw.get("metadata", raw) if isinstance(raw, dict) else {}
            color    = (m.get("dominant_color") or "").strip()
            category = (m.get("category")       or "").strip()
            if color and category:
                descriptions.append(f"{color} {category}")
            elif category:
                descriptions.append(category)
        if descriptions:
            parts.append(", ".join(descriptions))

    # 3. Styling techniques
    techniques = outfit_json.get("styling_techniques") or []
    if techniques:
        parts.append(", ".join(str(t) for t in techniques))

    # 4. Confidence mood
    confidence = (outfit_json.get("confidence") or "").strip()
    if confidence:
        parts.append(confidence)

    # 5. Editorial suffix (always last)
    parts.append(POSITIVE_SUFFIX)

    positive = ", ".join(p for p in parts if p)
    return positive, NEGATIVE_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_illustration(
    outfit_json: dict,
    item_metadata: Optional[list[dict]] = None,
    filename: str = "outfit_sd.png",
    steps: int = MAX_STEPS,
) -> str:
    """
    Run Stable Diffusion inference and save the result to ``output/``.

    This function is **blocking** (~20–40 s on MacBook Air).  Always call
    it from ``generate_illustration_async`` or a BackgroundTask so the
    FastAPI event loop stays free.

    Args:
        outfit_json:   Structured outfit dict (from Ollama).
        item_metadata: Optional ChromaDB metadata for richer SD prompt.
        filename:      Output filename inside ``output/``.
        steps:         Inference steps — hard-capped at ``MAX_STEPS`` (25).

    Returns:
        Absolute path to the saved PNG as a string.

    Raises:
        RuntimeError: If ``diffusers`` / ``torch`` are not installed.
    """
    import torch

    steps = min(steps, MAX_STEPS)          # hard cap — never exceed 25
    positive, negative = build_sd_prompt(outfit_json, item_metadata)
    logger.info("SD positive prompt: %s…", positive[:120])

    pipe   = _load_sd_pipeline()
    device = _get_device()

    try:
        with torch.no_grad():
            result = pipe(
                prompt=positive,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=GUIDANCE,
            )
        image = result.images[0]
    finally:
        # Always free MPS cache — even if inference raised an exception
        try:
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    out_path = OUTPUT_DIR / filename
    image.save(str(out_path), format="PNG")
    logger.info("SD illustration saved → %s", out_path)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Async / background wrappers
# ─────────────────────────────────────────────────────────────────────────────

async def generate_illustration_async(
    outfit_json: dict,
    item_metadata: Optional[list[dict]] = None,
    filename: str = "outfit_sd.png",
    steps: int = MAX_STEPS,
) -> str:
    """
    Non-blocking wrapper — runs ``generate_illustration`` in a thread-pool
    executor so the FastAPI event loop is never blocked during the 20–40 s
    inference window.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_illustration(outfit_json, item_metadata, filename, steps),
    )


def _run_generation_job(
    gen_id: str,
    outfit_json: dict,
    item_metadata: Optional[list[dict]],
    steps: int,
) -> None:
    """
    Blocking worker executed inside a FastAPI BackgroundTask thread.
    Updates ``_gen_jobs[gen_id]`` so the frontend can poll status.
    """
    _gen_jobs[gen_id]["status"] = "running"
    filename = f"outfit_sd_{gen_id}.png"
    try:
        path = generate_illustration(outfit_json, item_metadata, filename, steps)
        _gen_jobs[gen_id].update(status="done", path=path)
        logger.info("[%s] SD generation complete → %s", gen_id, path)
    except Exception as exc:
        logger.error("[%s] SD generation failed: %s", gen_id, exc, exc_info=True)
        _gen_jobs[gen_id].update(status="failed", error=str(exc))
    finally:
        # Evict oldest completed/failed jobs once cap is reached
        if len(_gen_jobs) > MAX_GEN_JOBS:
            evictable = [
                k for k, v in _gen_jobs.items()
                if v["status"] in ("done", "failed") and k != gen_id
            ]
            for k in evictable[: len(_gen_jobs) - MAX_GEN_JOBS]:
                del _gen_jobs[k]


def start_generation_job(
    outfit_json: dict,
    item_metadata: Optional[list[dict]] = None,
    steps: int = MAX_STEPS,
) -> str:
    """
    Register a generation job and return its ID immediately.

    The actual inference is run by ``_run_generation_job`` which must be
    submitted as a ``fastapi.BackgroundTasks`` task by the calling route.

    Returns:
        A unique ``gen_id`` string the frontend can poll at
        ``GET /styling/illustrate/{gen_id}``.
    """
    gen_id = uuid.uuid4().hex[:10]
    _gen_jobs[gen_id] = {"status": "queued", "path": None, "error": None}
    return gen_id


def get_generation_status(gen_id: str) -> Optional[dict]:
    """Return the current status dict for *gen_id*, or None if not found."""
    return _gen_jobs.get(gen_id)
