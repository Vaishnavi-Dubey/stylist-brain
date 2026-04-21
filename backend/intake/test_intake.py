"""
Quick sanity test for the rebuilt intake pipeline.

Usage:
    python -m backend.intake.test_intake /path/to/photo.jpg

Outputs:
    output/intake_test/<photo_stem>/
        cutout_<label>_NN.png        # individual RGBA garment cutouts
        _preview_grid.png             # all cutouts laid on a cream canvas
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from PIL import Image

import importlib.util

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_HERE = Path(__file__).parent
_detect  = _load("intake_detect",      _HERE / "detect.py")
_sam     = _load("intake_sam_refine",  _HERE / "sam_refine.py")
_segment = _load("intake_segment",     _HERE / "segment.py")
detect_garments = _detect.detect_garments
extract_all     = _segment.extract_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

CANVAS_BG = (245, 240, 235, 255)   # cream
TILE      = 360                    # px per cutout in the preview grid


def build_preview_grid(cutouts: list[Path], out_path: Path) -> Path:
    if not cutouts:
        raise RuntimeError("No cutouts produced — nothing to preview")

    n     = len(cutouts)
    cols  = min(n, 3)
    rows  = math.ceil(n / cols)
    pad   = 24
    label_h = 36
    canvas_w = cols * TILE + (cols + 1) * pad
    canvas_h = rows * (TILE + label_h) + (rows + 1) * pad
    canvas = Image.new("RGBA", (canvas_w, canvas_h), CANVAS_BG)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Futura.ttc", 20)
    except OSError:
        font = ImageFont.load_default()

    for i, p in enumerate(cutouts):
        r, c = divmod(i, cols)
        x0 = pad + c * (TILE + pad)
        y0 = pad + r * (TILE + label_h + pad)
        cut = Image.open(p).convert("RGBA")
        cut.thumbnail((TILE, TILE), Image.LANCZOS)
        cx = x0 + (TILE - cut.width)  // 2
        cy = y0 + (TILE - cut.height) // 2
        canvas.alpha_composite(cut, (cx, cy))
        label = p.stem.replace("cutout_", "")
        draw.text((x0 + TILE // 2, y0 + TILE + 6),
                  label, fill=(60, 50, 40, 255), anchor="ma", font=font)

    canvas.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    args = ap.parse_args()

    photo = args.photo.expanduser().resolve()
    if not photo.exists():
        print(f"❌ Not found: {photo}", file=sys.stderr)
        return 2

    out_dir = Path("output/intake_test") / photo.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▶ Test photo: {photo}")
    print(f"▶ Output dir: {out_dir}\n")

    detections = detect_garments(photo)
    if not detections:
        print("⚠ No garments detected.")
        return 1

    print("Detections:")
    for d in detections:
        print(f"  • {d.label:9s} <{d.raw_label:14s}>  bbox={d.bbox}  area={d.area_frac:.1%}")

    cutouts = extract_all(photo, detections, out_dir, prefix="cutout_")
    print(f"\n✔ Saved {len(cutouts)} cutouts to {out_dir}")

    grid = build_preview_grid(cutouts, out_dir / "_preview_grid.png")
    print(f"✔ Preview grid: {grid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
