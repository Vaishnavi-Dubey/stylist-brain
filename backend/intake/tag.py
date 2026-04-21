"""
tag.py — CLIP zero-shot garment tagger + dominant color analyser
Classifies a garment into a category and aesthetic vibes using the already-loaded
CLIP model. Piggybacks on embed.py's @lru_cache — no second model load.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Label sets ──────────────────────────────────────────────────────────────────

_CATEGORY_LABELS = [
    "t-shirt or casual top",
    "formal shirt or blouse",
    "dress or jumpsuit",
    "skirt",
    "trousers or chinos",
    "jeans or denim",
    "shorts",
    "jacket or blazer",
    "coat or overcoat",
    "sweater knitwear or hoodie",
    "shoes sneakers or boots",
    "handbag or purse",
    "scarf or shawl",
    "belt",
    "hat or cap",
    "jewellery or accessories",
]

# Maps the verbose CLIP label → simple Rule-of-Three category
_CATEGORY_MAP = {
    "t-shirt or casual top":        "top",
    "formal shirt or blouse":       "top",
    "dress or jumpsuit":            "dress",
    "skirt":                        "bottom",
    "trousers or chinos":           "bottom",
    "jeans or denim":               "bottom",
    "shorts":                       "bottom",
    "jacket or blazer":             "third-piece",
    "coat or overcoat":             "third-piece",
    "sweater knitwear or hoodie":   "top",
    "shoes sneakers or boots":      "shoes",
    "handbag or purse":             "third-piece",
    "scarf or shawl":               "third-piece",
    "belt":                         "third-piece",
    "hat or cap":                   "accessories",
    "jewellery or accessories":     "accessories",
}

_VIBE_LABELS = [
    "minimalist",
    "maximalist",
    "streetwear",
    "formal",
    "business casual",
    "smart casual",
    "casual everyday",
    "bohemian",
    "preppy",
    "athletic or sporty",
    "vintage or retro",
    "elegant or luxe",
    "edgy or grunge",
    "romantic or feminine",
    "utilitarian or workwear",
]

# ── Named color palette (RGB → name) ────────────────────────────────────────────

_COLOR_PALETTE: list[tuple[tuple[int, int, int], str]] = [
    ((0,   0,   0  ), "black"),
    ((20,  20,  20 ), "black"),
    ((50,  50,  50 ), "charcoal"),
    ((80,  80,  80 ), "dark grey"),
    ((128, 128, 128), "grey"),
    ((192, 192, 192), "light grey"),
    ((220, 220, 220), "off white"),
    ((255, 255, 255), "white"),
    ((255, 255, 240), "ivory"),
    ((245, 245, 220), "beige"),
    ((210, 180, 140), "tan"),
    ((245, 222, 179), "wheat"),
    ((210, 170, 109), "camel"),
    ((165, 120,  60 ), "cognac"),
    ((139,  90,  43 ), "brown"),
    ((165,  42,  42 ), "brown"),
    ((128,   0,   0 ), "maroon"),
    ((139,   0,   0 ), "dark red"),
    ((255,   0,   0 ), "red"),
    ((220,  20,  60 ), "crimson"),
    ((255, 105, 180), "hot pink"),
    ((255, 182, 193), "light pink"),
    ((255, 192, 203), "pink"),
    ((216, 112, 147), "blush"),
    ((128,   0, 128), "purple"),
    ((75,    0, 130 ), "indigo"),
    ((216, 191, 216), "lavender"),
    ((238, 130, 238), "violet"),
    ((0,    0, 139 ), "dark navy"),
    ((25,  25, 112 ), "midnight blue"),
    ((0,    0, 128 ), "navy"),
    ((0,    0, 205 ), "blue"),
    ((30, 144, 255 ), "dodger blue"),
    ((135, 206, 235), "sky blue"),
    ((173, 216, 230), "light blue"),
    ((0,  128, 128 ), "teal"),
    ((47,  79,  79 ), "dark teal"),
    ((64, 224, 208 ), "turquoise"),
    ((0,  100,   0 ), "dark green"),
    ((0,  128,   0 ), "green"),
    ((34, 139,  34 ), "forest green"),
    ((107, 142,  35), "olive"),
    ((144, 238, 144), "light green"),
    ((255, 255,   0), "yellow"),
    ((255, 215,   0), "gold"),
    ((255, 165,   0), "orange"),
    ((255, 140,   0), "dark orange"),
    ((112, 128, 144), "slate blue"),
]


# ── Cached text features ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _category_text_features():
    """Pre-compute CLIP text features for all category labels. Cached."""
    from intake.embed import _load_clip
    model, _, tokenizer = _load_clip()
    prompts = [f"a photo of {lbl}" for lbl in _CATEGORY_LABELS]
    tokens = tokenizer(prompts)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats   # shape: (n_labels, 512)


@lru_cache(maxsize=1)
def _vibe_text_features():
    """Pre-compute CLIP text features for all vibe labels. Cached."""
    from intake.embed import _load_clip
    model, _, tokenizer = _load_clip()
    prompts = [f"a {vibe} style fashion item" for vibe in _VIBE_LABELS]
    tokens = tokenizer(prompts)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def clear_tag_caches() -> None:
    """Call this alongside embed.unload_model() to free all CLIP-related RAM."""
    _category_text_features.cache_clear()
    _vibe_text_features.cache_clear()
    logger.info("Tag text-feature caches cleared")


# ── Public API ───────────────────────────────────────────────────────────────────

def classify_category(image_embedding: list[float]) -> str:
    """
    Zero-shot classify a garment into a simple Rule-of-Three category.

    Args:
        image_embedding: Pre-computed 512-d CLIP image embedding (from embed.py).

    Returns:
        One of: top, bottom, dress, third-piece, shoes, accessories
    """
    text_feats = _category_text_features()
    img_tensor = torch.tensor(image_embedding).unsqueeze(0)  # (1, 512)

    sims     = (img_tensor @ text_feats.T).squeeze()
    best_idx = int(sims.argmax())
    label    = _CATEGORY_LABELS[best_idx]

    category = _CATEGORY_MAP.get(label, "top")
    logger.debug("Category: %s → %s (sim=%.3f)", label, category, float(sims[best_idx]))
    return category


def classify_vibes(image_embedding: list[float], top_k: int = 3) -> list[str]:
    """
    Zero-shot assign the top-k aesthetic vibe tags to a garment.

    Args:
        image_embedding: Pre-computed 512-d CLIP image embedding.
        top_k:           Number of vibe tags to return.

    Returns:
        List of vibe tag strings sorted by relevance (highest first).
    """
    text_feats = _vibe_text_features()
    img_tensor = torch.tensor(image_embedding).unsqueeze(0)

    sims        = (img_tensor @ text_feats.T).squeeze()
    top_indices = sims.topk(min(top_k, len(_VIBE_LABELS))).indices.tolist()
    vibes       = [_VIBE_LABELS[i] for i in top_indices]

    logger.debug("Vibes: %s", vibes)
    return vibes


def dominant_color_name(image_path: str | Path) -> str:
    """
    Analyse pixel data to return a human-readable dominant color name.
    Uses K-means clustering (k=3) on garment pixels only.
    For RGBA images, transparent pixels (alpha < 128) are excluded.

    Args:
        image_path: Path to the segmented garment image (PNG or JPG).

    Returns:
        Color name string, e.g. "navy", "charcoal", "blush".
    """
    img  = Image.open(image_path).convert("RGBA")
    arr  = np.array(img)

    # Only sample pixels that are part of the actual garment
    mask   = arr[:, :, 3] > 128
    pixels = arr[mask][:, :3].astype(np.float32)

    if len(pixels) < 100:
        return "unknown"

    try:
        from sklearn.cluster import KMeans  # type: ignore
        k   = min(3, len(pixels))
        km  = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(pixels)
        dominant_idx = int(np.bincount(km.labels_).argmax())
        dominant_rgb = km.cluster_centers_[dominant_idx].astype(int)
    except ImportError:
        # sklearn not installed — filter extremes and fall back to median
        not_bg = (
            (pixels[:, 0] + pixels[:, 1] + pixels[:, 2]) > 50
        ) & (
            (pixels[:, 0] + pixels[:, 1] + pixels[:, 2]) < 720
        )
        filtered     = pixels[not_bg] if not_bg.sum() > 100 else pixels
        dominant_rgb = np.median(filtered, axis=0).astype(int)

    return _nearest_color_name(tuple(dominant_rgb))


def _nearest_color_name(rgb: tuple) -> str:
    r, g, b   = rgb
    best_name = "unknown"
    best_dist = float("inf")
    for (cr, cg, cb), name in _COLOR_PALETTE:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name
