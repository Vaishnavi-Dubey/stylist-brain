"""
segment.py — Per-garment RGBA cutout writer (rebuilt v2).

Takes a `GarmentDetection` from detect.py (which already carries a clean
per-pixel mask from SegFormer) and turns it into a tightly-cropped RGBA
PNG with a transparent background — ready for embed.py and the flat-lay
compositor.

Mask post-processing
--------------------
1. Morphological close → fill 1-pixel holes inside the garment.
2. Keep only the largest connected component → drop spurious specks.
3. Feather the edge by 1 px with a Gaussian blur on the alpha channel so
   the cutout doesn't look paper-cut against the cream flat-lay canvas.
4. Trim to bbox + 8 px breathing room.

Public surface
--------------
extract_garment_cutout(image_path, detection, output_path) -> Path | None
extract_all(image_path, detections, out_dir, prefix="") -> list[Path]
unload_model()  # no-op, kept for symmetry with detect.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageFilter

try:
    from .detect import GarmentDetection
    from . import sam_refine
except ImportError:
    from intake_detect import GarmentDetection  # standalone load (test_intake.py)
    import intake_sam_refine as sam_refine

logger = logging.getLogger(__name__)

_PAD                = 8     # px breathing room around the trimmed mask
_FEATHER_RADIUS     = 1.0   # px Gaussian blur on alpha edges
_MIN_OPAQUE_RATIO   = 0.04  # cutout must be ≥4 % opaque pixels to keep
_CLOSE_KERNEL_PX    = 3     # morphological close radius (fills small holes)


# ── Mask cleanup helpers ─────────────────────────────────────────────────────

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest 4-connected component."""
    from scipy.ndimage import label
    labelled, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background
    keep = int(sizes.argmax())
    return labelled == keep


def _morph_close(mask: np.ndarray, radius: int = _CLOSE_KERNEL_PX) -> np.ndarray:
    from scipy.ndimage import binary_closing, generate_binary_structure, iterate_structure
    struct = iterate_structure(generate_binary_structure(2, 1), radius)
    return binary_closing(mask, structure=struct)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    try:
        m = _morph_close(mask)
        m = _largest_component(m)
        return m
    except ImportError:
        logger.warning("scipy not installed — skipping mask cleanup")
        return mask


def _fill_holes_in_person(
    mask: np.ndarray,
    person_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    skin_veto: np.ndarray | None,
) -> np.ndarray:
    """Fill enclosed holes in *mask* using *person_mask* as a hard
    boundary. Hair strands lying on a shirt produce holes inside the
    garment; this recovers them as garment pixels without leaking
    outside the person silhouette.

    Holes are only filled if they are:
      • entirely inside the person silhouette,
      • inside the garment bbox (with a small margin),
      • NOT inside the skin/hair veto (so face/hair regions don't get
        labelled as garment).
    """
    if person_mask is None or not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure
    except ImportError:
        return mask

    H, W = mask.shape
    x1, y1, x2, y2 = bbox
    pad = 8
    bbox_mask = np.zeros_like(mask)
    bbox_mask[max(0, y1 - pad):min(H, y2 + pad + 1),
              max(0, x1 - pad):min(W, x2 + pad + 1)] = True

    # Dilate the mask slightly first so neighbouring single-pixel gaps
    # close into one connected region — improves binary_fill_holes hit rate.
    struct = generate_binary_structure(2, 2)
    grown   = binary_dilation(mask, structure=struct, iterations=2)
    filled  = binary_fill_holes(grown)
    candidate_fill = filled & ~grown                 # the gained holes
    candidate_fill &= person_mask & bbox_mask        # must stay inside person + bbox
    if skin_veto is not None:
        candidate_fill &= ~skin_veto                 # never recover hair/face/limbs

    if not candidate_fill.any():
        return mask
    out = mask | candidate_fill
    logger.debug("Hole-fill recovered %d px (inside person silhouette)",
                 int(candidate_fill.sum()))
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def extract_garment_cutout(
    image_path:  str | Path,
    detection:   GarmentDetection,
    output_path: str | Path,
    sam_predictor=None,
) -> Path | None:
    """Apply *detection.mask* to the source photo and save a tight RGBA PNG.

    If *sam_predictor* is supplied (created via
    :func:`sam_refine.make_predictor` for this photo), MobileSAM refines
    the SegFormer mask before cleanup and feathering.

    Returns the saved path, or ``None`` if the cutout failed the opacity
    quality gate.
    """
    image_path  = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(image_path).convert("RGB"))
    H, W = rgb.shape[:2]
    if detection.mask.shape != (H, W):
        logger.error("Mask shape %s != image %s", detection.mask.shape, (H, W))
        return None

    raw_mask = detection.mask
    if sam_predictor is not None:
        raw_mask, _ = sam_refine.refine(
            sam_predictor, raw_mask, detection.bbox,
            negative_mask=detection.sam_negative,
        )

    # Re-apply the skin/hair veto AFTER SAM — SAM tends to add hair/face
    # back into the mask because they sit inside the garment bbox and
    # belong to the same coherent silhouette from SAM's perspective.
    if detection.skin_veto is not None:
        raw_mask = raw_mask & ~detection.skin_veto

    # Fill internal holes (from removed hair strands) using the person
    # silhouette as a hard boundary. Anything inside the garment bbox
    # that is INSIDE the person and NOT covered by skin/hair veto is
    # plausibly part of the garment — recover it.
    if detection.person_mask is not None:
        raw_mask = _fill_holes_in_person(
            raw_mask, detection.person_mask, detection.bbox,
            detection.skin_veto,
        )

    mask = _clean_mask(raw_mask)

    # Build RGBA: original RGB + binary alpha
    alpha = (mask.astype(np.uint8) * 255)
    rgba = np.dstack([rgb, alpha])
    img_rgba = Image.fromarray(rgba, mode="RGBA")

    # Tight crop using the cleaned mask's bbox (recomputed — cleanup may shrink it)
    ys, xs = np.where(mask)
    if ys.size == 0:
        logger.debug("Empty mask after cleanup for %s", output_path.name)
        return None
    x1 = max(0, int(xs.min()) - _PAD)
    y1 = max(0, int(ys.min()) - _PAD)
    x2 = min(W, int(xs.max()) + _PAD)
    y2 = min(H, int(ys.max()) + _PAD)
    cropped = img_rgba.crop((x1, y1, x2, y2))

    # Feather the alpha edge so the cutout reads as natural fabric, not a sticker.
    if _FEATHER_RADIUS > 0:
        r, g, b, a = cropped.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=_FEATHER_RADIUS))
        cropped = Image.merge("RGBA", (r, g, b, a))

    # Quality gate
    arr = np.array(cropped)
    opaque_ratio = float((arr[:, :, 3] > 16).mean())
    if opaque_ratio < _MIN_OPAQUE_RATIO:
        logger.info("Skipping %s — only %.1f%% opaque",
                    output_path.name, opaque_ratio * 100)
        return None

    cropped.save(str(output_path), format="PNG", optimize=True)
    logger.info("Cutout → %s  (%dx%d, %.0f%% opaque, label=%s)",
                output_path.name, cropped.width, cropped.height,
                opaque_ratio * 100, detection.label)
    return output_path


def extract_all(
    image_path:  str | Path,
    detections:  Iterable[GarmentDetection],
    out_dir:     str | Path,
    prefix:      str = "",
    use_sam:     bool = True,
) -> List[Path]:
    """Write a cutout PNG per detection, returning the list of saved paths.

    When *use_sam* is True (default) we build a single MobileSAM predictor
    for the source photo and pass it to every cutout call — that way
    ``set_image`` runs once, not once per garment.

    Filenames: ``{prefix}{label}_{idx:02d}.png``  (idx disambiguates when
    the detector emits two items of the same canonical category).
    """
    image_path = Path(image_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_predictor = None
    if use_sam:
        try:
            rgb = np.array(Image.open(image_path).convert("RGB"))
            sam_predictor = sam_refine.make_predictor(rgb)
        except Exception as exc:
            logger.warning("MobileSAM unavailable (%s) — using SegFormer masks only", exc)
            sam_predictor = None

    saved: list[Path] = []
    seen_label_count: dict[str, int] = {}
    for det in detections:
        idx = seen_label_count.get(det.label, 0)
        seen_label_count[det.label] = idx + 1
        fname = f"{prefix}{det.label}_{idx:02d}.png"
        out  = extract_garment_cutout(
            image_path, det, out_dir / fname,
            sam_predictor=sam_predictor,
        )
        if out is not None:
            saved.append(out)
    return saved


def unload_model() -> None:
    """No model owned by this module — present for API symmetry."""
    pass
