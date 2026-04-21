"""
detect.py — Multi-garment semantic detector (rebuilt v2).

Single person photo in → list of garment masks out, one per detected
clothing item. Uses SegFormer-B2 fine-tuned on the ATR clothing dataset
(`mattmdjaga/segformer_b2_clothes`), which produces per-pixel labels for
17 classes including upper-clothes, pants, skirt, dress, jacket, bag,
shoes, hat, scarf.

Runs on Apple-Silicon MPS when available. The model + processor are
cached after first call so the cost is paid once at startup.

Public surface
--------------
detect_garments(image_path) -> list[GarmentDetection]
unload_model()

A GarmentDetection contains:
    label : canonical category ("top" | "bottom" | "dress" | "outerwear"
            | "shoes" | "bag" | "accessory")
    raw_label : the underlying SegFormer class name
    mask  : np.ndarray[bool] of shape (H, W) — full-image-sized garment mask
    bbox  : (x1, y1, x2, y2) tight bounding box of the mask in image coords
    area_frac : fraction of the image covered by the mask (used for ranking)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────────
_MODEL_ID = "sayeed99/segformer_b3_clothes"

# SegFormer-B2-Clothes class index → canonical garment category.
# Background / skin / hair are intentionally absent.
_CLASS_TO_CATEGORY: dict[int, tuple[str, str]] = {
    # idx : (raw_label,            canonical category)
    1:  ("hat",                    "accessory"),
    3:  ("sunglasses",             "accessory"),
    4:  ("upper-clothes",          "top"),
    5:  ("skirt",                  "bottom"),
    6:  ("pants",                  "bottom"),
    7:  ("dress",                  "dress"),
    8:  ("belt",                   "accessory"),
    9:  ("left-shoe",              "shoes"),
    10: ("right-shoe",             "shoes"),
    16: ("bag",                    "bag"),
    17: ("scarf",                  "accessory"),
}

# Hair / face leak the most into garment masks (the "shoulder horn"
# artifact) and need an aggressive dilation to stop SAM regrowing them.
_HAIR_FACE_CLASSES: tuple[int, ...] = (2, 11)
_HAIR_VETO_DILATE_PX = 18

# Arms / legs legitimately touch the garment edge — the veto must be
# narrow or sleeves and trouser edges get eaten.
_LIMB_CLASSES: tuple[int, ...] = (12, 13, 14, 15)
_LIMB_VETO_DILATE_PX = 8

# Backwards-compat alias for the diagnostic logging path.
_SKIN_HAIR_CLASSES = _HAIR_FACE_CLASSES + _LIMB_CLASSES
_VETO_DILATE_PX = _HAIR_VETO_DILATE_PX

# Logit-level fix for the "shoulder horns" artifact: dark hair flowing
# forward over the shoulders is frequently labeled as upper-clothes by
# SegFormer-B2-Clothes. We dilate the *confidently* hair/face region by
# this many pixels, then within that "influence zone" we subtract a
# penalty from the upper-clothes / dress / outerwear logits before
# argmax, so any borderline pixel flips back to hair/face.
_HAIR_INFLUENCE_PX     = 0      # disabled — opening handles horns better
_HAIR_LOGIT_PENALTY    = 0.0
_GARMENT_TORSO_CLASSES = (4, 7)   # Upper-clothes, Dress

# Morphological opening radius (px) applied to torso clothing masks (top
# / dress / outerwear). Removes thin hair-protrusions ("shoulder horns")
# while preserving the body of the garment. Tuned to be smaller than a
# typical shoulder strap so spaghetti tops aren't destroyed.
_TORSO_OPENING_PX = 6    # disconnect thin "horn" tendrils (hair/arm
                          # use color purification (below) instead.

# Color purification: within `_HAIR_PROXIMITY_PX` pixels of confident hair,
# any clothing-mask pixel whose color is closer to the hair-mean color
# than to the shirt-core color is dropped. Eliminates the "shoulder
# horns" leak that morphological tools cannot reach.
_HAIR_PROXIMITY_PX  = 180
_PURIFY_MIN_HAIR_PX = 500    # need this many confident hair px to trust the colour

# Outerwear (jackets/coats) are labelled "upper-clothes" by ATR — the
# distinction is made downstream by tag.py / embed.py based on visual
# features. We keep one "top" per photo and never split it.

# Categories that should be merged into a single mask when multiple class
# indices belong together (e.g. left-shoe + right-shoe → one "shoes" item).
_MERGE_CATEGORIES = {"shoes"}

_MIN_AREA_FRAC = 0.05   # ignore masks smaller than 5 % of the image
                         # (keeps fragmented detections out of the wardrobe)
_DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class GarmentDetection:
    label:     str
    raw_label: str
    mask:      np.ndarray            # bool (H, W) — already veto-subtracted
    bbox:      tuple[int, int, int, int]
    area_frac: float
    skin_veto:    np.ndarray | None = None   # bool (H, W) — pixels that must
                                             #   stay transparent even if SAM
                                             #   tries to re-include them
    sam_negative: np.ndarray | None = None   # bool (H, W) — confident hair /
                                             #   face / skin pixels passed to
                                             #   SAM as negative point prompts
    person_mask:  np.ndarray | None = None   # bool (H, W) — full person
                                             #   silhouette from rembg, used
                                             #   as a hole-filling boundary

    def __repr__(self) -> str:
        return (
            f"GarmentDetection({self.label!r} <{self.raw_label}> "
            f"bbox={self.bbox} area={self.area_frac:.1%})"
        )


# ── Lazy model loader ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    logger.info("Loading SegFormer-B3-Clothes on %s …", _DEVICE)
    processor = SegformerImageProcessor.from_pretrained(_MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_ID).to(_DEVICE)
    model.eval()
    return processor, model


def _person_silhouette(pil_img: Image.Image) -> np.ndarray | None:
    """Return a bool mask of the full person silhouette using rembg's
    u2net_human_seg model. Used as a hole-filling boundary so we can
    fill hair-on-shirt holes without leaking outside the body.

    Returns ``None`` if rembg is unavailable.
    """
    try:
        from rembg import new_session, remove
    except ImportError:
        logger.info("rembg not installed — skipping person silhouette")
        return None
    try:
        session = _rembg_session()
        rgba = remove(pil_img, session=session)
        alpha = np.array(rgba)[:, :, 3]
        return alpha > 64
    except Exception as exc:
        logger.warning("rembg failed: %s — skipping person silhouette", exc)
        return None


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session
    logger.info("Loading rembg u2net_human_seg session …")
    return new_session("u2net_human_seg")


# ── Public API ───────────────────────────────────────────────────────────────

def detect_garments(image_path: str | Path) -> List[GarmentDetection]:
    """Detect every garment in a single person photo.

    Returns a list of GarmentDetection objects sorted by area (largest
    first). Tops and bottoms are kept whole; left/right shoes are merged
    into a single "shoes" detection.
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    processor, model = _load_model()

    inputs = processor(images=img, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, C, h, w) — h,w small
    upsampled = F.interpolate(
        logits, size=(H, W), mode="bilinear", align_corners=False
    )

    # ── Logit-level "shoulder horn" fix ─────────────────────────────────────
    # Identify a confident hair/face region from a first-pass argmax, dilate
    # it, and within that influence zone subtract a penalty from torso-
    # clothing logits so borderline pixels flip back to hair/face.
    pred0 = upsampled.argmax(dim=1)[0].cpu().numpy()
    hair_face_seed = np.isin(pred0, (2, 11))    # only the truly confident pixels
    if hair_face_seed.any() and _HAIR_INFLUENCE_PX > 0:
        try:
            from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
            struct = iterate_structure(generate_binary_structure(2, 1), _HAIR_INFLUENCE_PX)
            influence = binary_dilation(hair_face_seed, structure=struct)
        except ImportError:
            influence = hair_face_seed
        if influence.any():
            penalty_mask = torch.from_numpy(influence).to(_DEVICE)
            for cls_id in _GARMENT_TORSO_CLASSES:
                upsampled[0, cls_id][penalty_mask] -= _HAIR_LOGIT_PENALTY

    pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    # Numpy view of the source photo, used by color-based purification.
    rgb_arr = np.array(img)
    hair_mask = pred == 2

    # Confident skin/hair pixels — passed to MobileSAM downstream as
    # negative point prompts so SAM doesn't include them in the refined
    # garment mask. Built from the raw argmax (no dilation) since SAM's
    # point prompts only need pixels that are definitely-NOT-the-object.
    sam_negative = np.isin(pred, _HAIR_FACE_CLASSES + _LIMB_CLASSES)

    # Person silhouette via rembg — used downstream as a hard boundary
    # for hole-filling (so hair-on-shirt holes can be filled without
    # leaking outside the person's body). Lazy imported.
    person_mask = _person_silhouette(img)

    # Build the skin/hair veto. We keep two: a wide veto for hair/face
    # (kills shoulder-horn leaks) and a narrow veto for limbs (just enough
    # to clean the rim without eating sleeves or trouser edges).
    veto = np.zeros_like(pred, dtype=bool)
    try:
        from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
        for cls_set, radius in (
            (_HAIR_FACE_CLASSES, _HAIR_VETO_DILATE_PX),
            (_LIMB_CLASSES,      _LIMB_VETO_DILATE_PX),
        ):
            sub = np.isin(pred, cls_set)
            if sub.any() and radius > 0:
                struct = iterate_structure(generate_binary_structure(2, 1), radius)
                sub = binary_dilation(sub, structure=struct)
            veto |= sub
    except ImportError:
        veto = np.isin(pred, _SKIN_HAIR_CLASSES)
        logger.warning("scipy missing — skin/hair veto used un-dilated")

    detections: list[GarmentDetection] = []
    image_area = float(W * H)

    # Group raw class indices by canonical category so paired items merge.
    grouped: dict[str, list[tuple[int, str]]] = {}
    for cls_id, (raw, cat) in _CLASS_TO_CATEGORY.items():
        grouped.setdefault(cat, []).append((cls_id, raw))

    for category, members in grouped.items():
        if category in _MERGE_CATEGORIES or len(members) == 1:
            mask = np.zeros_like(pred, dtype=bool)
            raws = []
            for cls_id, raw in members:
                m = pred == cls_id
                if m.any():
                    mask |= m
                    raws.append(raw)
            if not mask.any():
                continue
            mask &= ~veto
            local_veto = veto
            if category in ("top", "dress", "outerwear"):
                pre_purify = mask.copy()
                mask = _open_mask(mask, _TORSO_OPENING_PX)
                mask = _color_purify_against_hair(rgb_arr, mask, hair_mask)
                local_veto = veto | (pre_purify & ~mask)
            if not mask.any():
                continue
            det = _build_detection(category, "+".join(raws) or members[0][1],
                                   mask, image_area, local_veto,
                                   sam_negative, person_mask)
            if det is not None:
                detections.append(det)
        else:
            for cls_id, raw in members:
                m = pred == cls_id
                if not m.any():
                    continue
                m &= ~veto
                if not m.any():
                    continue
                det = _build_detection(category, raw, m, image_area, veto,
                                       sam_negative, person_mask)
                if det is not None:
                    detections.append(det)

    detections.sort(key=lambda d: d.area_frac, reverse=True)
    logger.info("Detected %d garment(s) in %s: %s",
                len(detections), image_path.name,
                [(d.label, f"{d.area_frac:.1%}") for d in detections])
    return detections


def _build_detection(
    label: str, raw: str, mask: np.ndarray, image_area: float,
    veto: np.ndarray | None = None,
    sam_negative: np.ndarray | None = None,
    person_mask:  np.ndarray | None = None,
) -> GarmentDetection | None:
    area = float(mask.sum())
    frac = area / image_area
    if frac < _MIN_AREA_FRAC:
        return None
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return GarmentDetection(label=label, raw_label=raw, mask=mask,
                            bbox=bbox, area_frac=frac, skin_veto=veto,
                            sam_negative=sam_negative, person_mask=person_mask)


def _open_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening (erode-then-dilate). Removes thin protrusions."""
    if radius <= 0 or not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_opening, generate_binary_structure, iterate_structure
    except ImportError:
        return mask
    struct = iterate_structure(generate_binary_structure(2, 1), radius)
    return binary_opening(mask, structure=struct)


def _color_purify_against_hair(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    hair_mask: np.ndarray,
    proximity_px: int = _HAIR_PROXIMITY_PX,
) -> np.ndarray:
    """Drop mask pixels near confident hair pixels whose color is closer
    to the hair-mean than to the shirt-core mean.

    This rescues SegFormer's "shoulder horns" failure mode where dark
    hair flowing forward over the shoulders is mis-labeled as
    upper-clothes. SegFormer only confidently labels the *outer* hair as
    Hair, so a class-only veto can't help; but those horn pixels are
    visually identical to the rest of the hair, so a color test does.
    """
    if hair_mask.sum() < _PURIFY_MIN_HAIR_PX or not mask.any():
        return mask
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return mask

    dist_to_hair = distance_transform_edt(~hair_mask)
    in_zone = mask & (dist_to_hair <= proximity_px)
    if not in_zone.any():
        return mask

    rgb = rgb_image.astype(np.float32)
    hair_color = rgb[hair_mask].mean(axis=0)

    shirt_core = mask & (dist_to_hair > proximity_px)
    shirt_color = (
        rgb[shirt_core].mean(axis=0) if shirt_core.any()
        else rgb[mask].mean(axis=0)
    )

    if np.linalg.norm(hair_color - shirt_color) < 25:
        # Hair colour ≈ shirt colour (e.g. dark hair + black shirt) — color
        # purification would erase legitimate shirt pixels. Bail out.
        logger.info("Color purify skipped — hair/shirt colours too similar "
                    "(Δ=%.1f)", float(np.linalg.norm(hair_color - shirt_color)))
        return mask

    zone_pixels = rgb[in_zone]
    d_hair  = np.linalg.norm(zone_pixels - hair_color,  axis=1)
    d_shirt = np.linalg.norm(zone_pixels - shirt_color, axis=1)
    drop = d_hair < d_shirt

    new_mask = mask.copy()
    ys, xs = np.where(in_zone)
    new_mask[ys[drop], xs[drop]] = False

    # The dropped pixels are often individual hair strands lying on top
    # of the shirt — visually correct to remove, but they leave a "swiss
    # cheese" pattern of small holes inside an otherwise solid mask.
    # Close holes that are well INSIDE the shirt (i.e. far from the
    # original hair region) to fix this without re-growing the horns.
    try:
        from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, iterate_structure, label
        struct = iterate_structure(generate_binary_structure(2, 1), 6)
        closed = binary_closing(new_mask, structure=struct)
        # Only re-apply the closing where the original mask was True AND
        # we are far from hair (so we never reincorporate the horns).
        safe_zone = mask & (dist_to_hair > proximity_px / 2)
        new_mask = new_mask | (closed & safe_zone)

        # Break thin tendrils (residual hair strands hanging off the
        # shoulder by a narrow connection) without damaging the body.
        # Erode/dilate by 10 px → snaps any bridge thinner than 20 px,
        # then we keep only the connected component containing the
        # largest mass (= the actual garment).
        thin_struct = iterate_structure(generate_binary_structure(2, 1), 10)
        snapped = binary_opening(new_mask, structure=thin_struct)
        labelled, n = label(snapped)
        if n >= 1:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            keep = int(sizes.argmax())
            core = labelled == keep
            # Re-include shirt pixels in the original mask that touch the
            # core via a NORMAL (not thin) bridge — done by dilating the
            # core back and intersecting with the un-opened mask.
            core_grown = np.zeros_like(core)
            from scipy.ndimage import binary_dilation
            core_grown = binary_dilation(core, structure=thin_struct)
            new_mask = new_mask & core_grown
    except ImportError:
        pass

    logger.info("Color purify dropped %d/%d zone pixels (hair=%s shirt=%s)",
                int(drop.sum()), int(in_zone.sum()),
                hair_color.round(0).astype(int).tolist(),
                shirt_color.round(0).astype(int).tolist())
    return new_mask


def unload_model() -> None:
    _load_model.cache_clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("SegFormer-B2-Clothes unloaded")
