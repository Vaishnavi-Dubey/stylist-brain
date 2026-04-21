"""
embed.py — CLIP style embedding generator
Converts a segmented garment image into a 512-d style vector.

Model: CLIP ViT-B/32 via open_clip — lightweight, CPU-friendly.
⚠️  Do NOT use ViT-L/14 — too large for MacBook Air RAM budget.

Pipeline position: step 4 of 4 (video → detect → segment → embed)
"""

import logging
from pathlib import Path
from functools import lru_cache

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED  = "openai"   # weights released by OpenAI under MIT licence


@lru_cache(maxsize=1)
def _load_clip():
    """Lazy-load CLIP model + transforms. Cached after first call."""
    import open_clip  # type: ignore

    logger.info("Loading CLIP %s (%s)...", CLIP_MODEL_NAME, CLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer


def embed_image(image_path: str | Path) -> list[float]:
    """
    Generate a 512-d CLIP embedding for a garment image.

    Args:
        image_path: Path to a segmented garment PNG/JPG.

    Returns:
        L2-normalised embedding as a plain Python list[float].
    """
    model, preprocess, _ = _load_clip()
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)   # L2 normalise

    return features.squeeze().tolist()


def embed_text(query: str) -> list[float]:
    """
    Generate a 512-d CLIP embedding for a natural-language style query.

    Args:
        query: e.g. "sharp but approachable for a rainy Monday meeting"

    Returns:
        L2-normalised embedding as a plain Python list[float].
    """
    import open_clip  # type: ignore

    model, _, tokenizer = _load_clip()
    tokens = tokenizer([query])

    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features.squeeze().tolist()


def unload_model() -> None:
    """Free CLIP RAM and all tag text-feature caches. Call after batch intake completes."""
    _load_clip.cache_clear()
    try:
        from intake.tag import clear_tag_caches
        clear_tag_caches()
    except ImportError:
        pass
    logger.info("CLIP + tag caches unloaded")
