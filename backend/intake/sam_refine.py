"""
sam_refine.py — MobileSAM mask refinement for SegFormer detections.

SegFormer-B2-Clothes gives semantically correct but pixel-jagged garment
masks (especially around hair/shoulder/neckline boundaries). MobileSAM,
prompted by SegFormer's mask, returns a much sharper boundary because it
was trained on edge-aware promptable segmentation.

Workflow per source photo:
  predictor = make_predictor(rgb_array)        # set_image() runs once
  refined_mask = refine(predictor, segformer_mask, bbox)

If MobileSAM disagrees too much with SegFormer (low IoU or the chosen
mask shrinks/explodes), we keep SegFormer's mask — silent failure on
refinement is always better than a garbage mask going into the flat-lay.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SAM_CHECKPOINT = Path(__file__).parents[2] / "models" / "mobile_sam.pt"
_DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

_MIN_ACCEPT_IOU      = 0.55   # refined vs original IoU below this → reject
_MAX_AREA_RATIO      = 1.6    # refined mask >1.6× original area → reject (likely leaked)
_MIN_AREA_RATIO      = 0.5    # refined mask <0.5× original area → reject (likely shrunk)
_FG_POINT_COUNT      = 8      # foreground prompt points sampled from mask interior
_BG_POINT_COUNT      = 4      # background prompt points sampled outside bbox
_NEG_POINT_COUNT     = 8      # negative prompt points sampled from skin/hair INSIDE bbox


# ── Lazy SAM loader ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sam():
    from mobile_sam import sam_model_registry, SamPredictor
    if not _SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"MobileSAM weights missing: {_SAM_CHECKPOINT}")
    logger.info("Loading MobileSAM (vit_t) on %s …", _DEVICE)
    sam = sam_model_registry["vit_t"](checkpoint=str(_SAM_CHECKPOINT))
    sam.to(_DEVICE)
    sam.eval()
    return SamPredictor(sam)


def make_predictor(rgb: np.ndarray):
    """Return a SamPredictor with the given image already loaded.

    Call this **once per source photo** and pass the predictor to every
    refine() call for that photo — set_image() is the expensive step.
    """
    predictor = _load_sam()
    predictor.set_image(rgb)
    return predictor


# ── Prompt sampling ──────────────────────────────────────────────────────────

def _sample_points(mask: np.ndarray, bbox: tuple[int, int, int, int],
                   negative_mask: np.ndarray | None = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Build (point_coords, point_labels) prompts for SAM.

    Foreground points are sampled from inside the eroded garment mask.
    Negative points come from two sources:
      • *negative_mask* pixels that fall inside the bbox (e.g. confident
        hair / face / arm pixels that SAM would otherwise include).
      • Random scatter outside the bbox (background context).
    """
    H, W = mask.shape
    rng = np.random.default_rng(0)
    x1, y1, x2, y2 = bbox

    # Foreground: erode mask first so points are well inside the garment.
    try:
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(mask, iterations=4)
        if not interior.any():
            interior = mask
    except ImportError:
        interior = mask

    fg_ys, fg_xs = np.where(interior)
    if fg_ys.size == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=int)
    n_fg  = min(_FG_POINT_COUNT, fg_ys.size)
    pick  = rng.choice(fg_ys.size, size=n_fg, replace=False)
    fg_pts = np.stack([fg_xs[pick], fg_ys[pick]], axis=1)

    # Negative #1: pixels of confident hair/face/skin that fall inside
    # the garment bbox — tells SAM "even though these are inside the box,
    # they aren't part of the object."
    bg_inside_pts: list[tuple[int, int]] = []
    if negative_mask is not None:
        in_bbox = np.zeros_like(negative_mask, dtype=bool)
        in_bbox[y1:y2 + 1, x1:x2 + 1] = True
        neg_in_bbox = negative_mask & in_bbox
        nys, nxs = np.where(neg_in_bbox)
        if nys.size:
            n_neg = min(_NEG_POINT_COUNT, nys.size)
            sel   = rng.choice(nys.size, size=n_neg, replace=False)
            bg_inside_pts = list(zip(nxs[sel].tolist(), nys[sel].tolist()))

    # Negative #2: random points strictly outside the bbox.
    bg_outside_pts: list[tuple[int, int]] = []
    margin = 12
    attempts = 0
    while len(bg_outside_pts) < _BG_POINT_COUNT and attempts < 200:
        attempts += 1
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H))
        if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
            continue
        if mask[y, x]:
            continue
        bg_outside_pts.append((x, y))

    bg_arr = np.array(bg_inside_pts + bg_outside_pts, dtype=int) \
        if (bg_inside_pts or bg_outside_pts) else np.empty((0, 2), dtype=int)

    coords = np.concatenate([fg_pts, bg_arr], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(fg_pts), dtype=int),
        np.zeros(len(bg_arr), dtype=int),
    ])
    return coords, labels


# ── Refinement ───────────────────────────────────────────────────────────────

def refine(predictor, mask: np.ndarray,
           bbox: tuple[int, int, int, int],
           negative_mask: np.ndarray | None = None,
           ) -> tuple[np.ndarray, bool]:
    """Refine *mask* using MobileSAM. Returns (refined_or_original, was_refined).

    *negative_mask* (optional) is a bool array marking pixels that
    definitely should NOT be part of the object — typically the union of
    confident hair/face/skin pixels. SAM samples negative point prompts
    from inside the bbox where this mask is True, which is the proper
    fix for hair/face leaking into the refined garment mask.
    """
    if not mask.any():
        return mask, False

    coords, labels = _sample_points(mask, bbox, negative_mask=negative_mask)
    box_arr = np.array(bbox, dtype=np.float32)

    try:
        masks, scores, _ = predictor.predict(
            point_coords=coords if len(coords) else None,
            point_labels=labels if len(labels) else None,
            box=box_arr,
            multimask_output=True,
        )
    except Exception as exc:
        logger.warning("MobileSAM predict failed: %s — keeping SegFormer mask", exc)
        return mask, False

    orig_area = float(mask.sum())
    best_idx, best_iou = -1, -1.0
    for i, m in enumerate(masks):
        inter = float(np.logical_and(m, mask).sum())
        union = float(np.logical_or(m, mask).sum())
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, i

    if best_idx < 0 or best_iou < _MIN_ACCEPT_IOU:
        logger.info("SAM refine rejected (best IoU=%.2f < %.2f) — keeping SegFormer",
                    best_iou, _MIN_ACCEPT_IOU)
        return mask, False

    chosen = masks[best_idx].astype(bool)
    ratio  = float(chosen.sum()) / max(orig_area, 1.0)
    if not (_MIN_AREA_RATIO <= ratio <= _MAX_AREA_RATIO):
        logger.info("SAM refine rejected (area ratio=%.2f) — keeping SegFormer", ratio)
        return mask, False

    logger.info("SAM refine accepted (IoU=%.2f, area ratio=%.2f, score=%.2f)",
                best_iou, ratio, float(scores[best_idx]))
    return chosen, True


def unload_model() -> None:
    _load_sam.cache_clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("MobileSAM unloaded")
