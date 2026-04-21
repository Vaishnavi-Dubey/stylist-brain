"""
flatlay.py — Pillow-based flat-lay composite image generator (Mode 1).

Composes cropped wardrobe item images side-by-side on a dark canvas and
overlays category badges + a styling-technique label.  Uses only Pillow —
no ML inference, no model loading — so it is fast enough to run synchronously
within a FastAPI route without blocking the event loop for more than ~300ms.

Canvas layout
─────────────
  [PADDING] [SLOT_W] [GAP] [SLOT_W] [GAP] ... [SLOT_W] [PADDING]
  ↑ PADDING
  ┌────────────────────────────────────┐  ↑
  │          item image (centred)      │  SLOT_H
  │                                    │  ↓
  │         [CATEGORY BADGE]          │
  └────────────────────────────────────┘
                [TECHNIQUE LABEL]        ← FOOTER_H
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Layout ────────────────────────────────────────────────────────────────────
SLOT_W    = 300          # px width of each item tile
SLOT_H    = 400          # px height of each item tile
GAP       = 10           # horizontal gap between tiles
PADDING   = 10           # outer left/right/top margin
FOOTER_H  = 50           # space below tiles for technique label
CANVAS_H  = SLOT_H + PADDING * 2 + FOOTER_H

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COLOR  = (18, 18, 18)
GOLD      = (201, 168, 76)
WHITE     = (255, 255, 255)
BADGE_BG  = (40, 40, 40, 200)  # semi-transparent dark pill

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parents[2]
OUTPUT_DIR   = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Positional fallback labels when metadata category is missing ──────────────
_FALLBACK_LABELS = ["TOP", "BOTTOM", "THIRD-PIECE"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try common macOS / Linux font paths; fall back to Pillow built-in."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _fit_image(img: Image.Image, w: int, h: int) -> Image.Image:
    """
    Resize *img* to fit within a *w* × *h* slot while preserving aspect ratio.
    Returns a new RGBA canvas with an opaque dark background so all slots are
    the same visible size regardless of item image dimensions.
    """
    img.thumbnail((w, h), Image.LANCZOS)
    # Opaque dark background (same as canvas BG) so every slot is uniform 300×400
    canvas = Image.new("RGBA", (w, h), (18, 18, 18, 255))
    x = (w - img.width) // 2
    y = (h - img.height) // 2
    if img.mode == "RGBA":
        canvas.paste(img, (x, y), mask=img.split()[3])
    else:
        canvas.paste(img.convert("RGBA"), (x, y))
    return canvas


def _draw_badge(
    draw: ImageDraw.Draw,
    text: str,
    cx: int,
    cy: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw a pill-shaped semi-transparent badge centred on (cx, cy)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    draw.rounded_rectangle(
        [cx - tw // 2 - pad, cy - th // 2 - pad,
         cx + tw // 2 + pad, cy + th // 2 + pad],
        radius=4,
        fill=BADGE_BG,
    )
    draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=WHITE)


def _draw_placeholder(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    item_id: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw a muted placeholder when an item image cannot be loaded."""
    draw.rectangle([x, y, x + SLOT_W, y + SLOT_H], outline=(60, 60, 60), width=1)
    label = f"[{item_id[-6:]}]"
    bbox  = draw.textbbox((0, 0), label, font=font)
    tw    = bbox[2] - bbox[0]
    draw.text(
        (x + (SLOT_W - tw) // 2, y + SLOT_H // 2),
        label,
        font=font,
        fill=(80, 80, 80),
    )


def _resolve_image_path(item_id: str, meta: Optional[dict]) -> Optional[Path]:
    """
    Return the best available image path for an item, in priority order:
      1. ``image_path`` stored in ChromaDB / SQLite metadata (absolute path).
      2. SQLite wardrobe_items lookup.
      3. Filesystem rglob through wardrobe/ tree.
    """
    # 1. Metadata shortcut (fast path — always prefer this)
    if meta:
        raw_path = meta.get("image_path") or meta.get("metadata", {}).get("image_path")
        if raw_path:
            p = Path(raw_path)
            if p.exists():
                return p

    # 2. SQLite lookup
    try:
        from db.sqlite import get_connection
        with get_connection() as conn:
            row = conn.execute(
                "SELECT image_path FROM wardrobe_items WHERE id = ?", (item_id,)
            ).fetchone()
        if row and row["image_path"]:
            p = Path(row["image_path"])
            if p.exists():
                return p
    except Exception as exc:
        logger.debug("SQLite lookup for %s failed: %s", item_id, exc)

    # 3. Filesystem search (slow but catches any layout)
    wardrobe_dir = _ROOT / "wardrobe"
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for candidate in wardrobe_dir.rglob(f"*{item_id}*{ext}"):
            return candidate

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_flatlay(
    item_ids: list[str],
    technique: str,
    item_metadata: Optional[list[dict]] = None,
    filename: str = "outfit_preview.png",
) -> str:
    """
    Generate a flat-lay composite PNG for the given outfit item IDs.

    Args:
        item_ids:      Ordered item IDs (top → bottom → third-piece).
        technique:     Styling technique, e.g. ``"french tuck"``.
        item_metadata: List of metadata dicts parallel to *item_ids*.
                       Each dict may contain ``image_path`` and ``category``.
                       Accepts both flat ``{"image_path": ..., "category": ...}``
                       and nested ``{"metadata": {"image_path": ..., ...}}`` shapes.
        filename:      Output filename inside ``output/``.

    Returns:
        Absolute path to the saved PNG as a string.

    Raises:
        ValueError: If *item_ids* is empty.
    """
    if not item_ids:
        raise ValueError("create_flatlay: item_ids must not be empty.")

    n         = len(item_ids)
    canvas_w  = PADDING + n * SLOT_W + (n - 1) * GAP + PADDING
    canvas    = Image.new("RGB", (canvas_w, CANVAS_H), BG_COLOR)
    draw      = ImageDraw.Draw(canvas)

    badge_font     = _load_font(12)
    technique_font = _load_font(14)
    placeholder_font = _load_font(11)

    # ── Flatten metadata for easy lookup ─────────────────────────────────────
    meta_list: list[Optional[dict]] = []
    for i in range(n):
        if item_metadata and i < len(item_metadata):
            raw = item_metadata[i]
            # Normalise nested shape {"metadata": {...}} → flat
            if "metadata" in raw and isinstance(raw["metadata"], dict):
                meta_list.append(raw["metadata"])
            else:
                meta_list.append(raw)
        else:
            meta_list.append(None)

    # ── Composite each item ───────────────────────────────────────────────────
    for idx, item_id in enumerate(item_ids):
        slot_x = PADDING + idx * (SLOT_W + GAP)
        slot_y = PADDING
        meta   = meta_list[idx]

        img_path = _resolve_image_path(item_id, meta)
        if img_path:
            try:
                img    = Image.open(img_path)
                fitted = _fit_image(img, SLOT_W, SLOT_H)
                # fitted is always RGBA with opaque dark background — paste directly
                canvas.paste(fitted.convert("RGB"), (slot_x, slot_y))
            except Exception as exc:
                logger.warning("Could not load image %s: %s — placeholder", img_path, exc)
                _draw_placeholder(draw, slot_x, slot_y, item_id, placeholder_font)
        else:
            logger.warning("No image found for %s — using placeholder", item_id)
            _draw_placeholder(draw, slot_x, slot_y, item_id, placeholder_font)

        # ── Category badge ────────────────────────────────────────────────────
        category = ""
        if meta:
            category = (meta.get("category") or "").upper()
        if not category:
            category = _FALLBACK_LABELS[idx] if idx < len(_FALLBACK_LABELS) else f"ITEM {idx + 1}"

        _draw_badge(
            draw,
            category,
            cx=slot_x + SLOT_W // 2,
            cy=slot_y + SLOT_H - 18,
            font=badge_font,
        )

    # ── Technique label (centred in footer) ───────────────────────────────────
    if technique:
        label = technique.upper()
        bbox  = draw.textbbox((0, 0), label, font=technique_font)
        tw    = bbox[2] - bbox[0]
        tx    = (canvas_w - tw) // 2
        ty    = SLOT_H + PADDING * 2 + (FOOTER_H - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), label, font=technique_font, fill=GOLD)

    out_path = OUTPUT_DIR / filename
    canvas.save(str(out_path), format="PNG", optimize=True)
    logger.info("Flat-lay saved → %s  (%d item(s))", out_path, n)
    return str(out_path)


async def create_flatlay_async(
    item_ids: list[str],
    technique: str,
    item_metadata: Optional[list[dict]] = None,
    filename: str = "outfit_preview.png",
) -> str:
    """Thread-pool wrapper so create_flatlay never blocks the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: create_flatlay(item_ids, technique, item_metadata, filename),
    )
