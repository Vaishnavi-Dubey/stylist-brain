"""
pipeline.py — Wardrobe intake orchestrator (rebuilt for SegFormer-B3 +
MobileSAM multi-garment single-photo flow).

For each uploaded photo:
  1.  Detect every garment in one SegFormer pass (top, bottom, dress,
      outerwear, bag, shoes, etc.).
  2.  Refine each detection with MobileSAM (one set_image per photo).
  3.  Write a clean RGBA cutout PNG per garment.
  4.  Embed (CLIP) + tag (Ollama) + store (ChromaDB + SQLite).

RAM strategy (MacBook Air safe):
  • SegFormer-B3 + rembg + MobileSAM stay loaded for the duration of
    Phase 2 — they are small (~600 MB combined) and re-loading per photo
    would cost more than keeping them resident.
  • CLIP loads only for Phase 3, then unloads.
  • SD pipeline never loads here (only `imaging/enhance.py`).
"""

import json
import logging
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory job registry.  { job_id: { status, items_added, total, error } }
_jobs: dict[str, dict] = {}
MAX_JOBS = 50

WARDROBE_DIR = Path(__file__).parents[2] / "wardrobe"

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def run_pipeline(job_id: str, file_paths: list[Path]) -> None:
    """Full intake pipeline — runs as a FastAPI BackgroundTask."""
    job_dir     = WARDROBE_DIR / job_id
    frames_dir  = job_dir / "frames"
    cutouts_dir = job_dir / "cutouts"

    _set(job_id, status="processing", items_added=0, total=0, error=None)

    try:
        # ── Phase 1: collect source photos ─────────────────────────────
        source_photos: list[Path] = []
        for path in file_paths:
            if path.suffix.lower() in _VIDEO_EXTS:
                from intake.video import extract_frames
                extracted = extract_frames(path, frames_dir)
                source_photos.extend(extracted)
                logger.info("[%s] Extracted %d frames from %s",
                            job_id, len(extracted), path.name)
            elif path.suffix.lower() in _IMAGE_EXTS:
                source_photos.append(path)
            else:
                logger.warning("[%s] Skipping unsupported file: %s",
                               job_id, path.name)

        if not source_photos:
            raise ValueError("No valid images found in upload.")

        _set(job_id, total=len(source_photos))
        logger.info("[%s] Phase 1 done — %d source photo(s)",
                    job_id, len(source_photos))

        # ── Phase 2: detect + segment per photo ────────────────────────
        cutouts_dir.mkdir(parents=True, exist_ok=True)
        all_cutouts: list[tuple[str, Path]] = []   # (label, path)

        try:
            from intake.detect  import detect_garments, unload_model as _unload_seg
            from intake.segment import extract_all

            for photo in source_photos:
                try:
                    detections = detect_garments(photo)
                except Exception as exc:
                    logger.warning("[%s] Detection failed for %s: %s — skipping",
                                   job_id, photo.name, exc)
                    continue

                if not detections:
                    logger.info("[%s] No garments detected in %s",
                                job_id, photo.name)
                    continue

                logger.info("[%s] %s → %s",
                            job_id, photo.name,
                            [(d.label, f"{d.area_frac*100:.1f}%") for d in detections])

                stem = f"{photo.stem}_"
                try:
                    saved_paths = extract_all(
                        photo, detections, cutouts_dir,
                        prefix=stem, use_sam=True,
                    )
                except Exception as exc:
                    logger.error("[%s] Segmentation failed for %s: %s",
                                 job_id, photo.name, exc, exc_info=True)
                    continue

                # extract_all returns paths in detection order — re-zip with labels
                for det, p in zip(detections, saved_paths):
                    all_cutouts.append((det.label, p))
                    logger.info("[%s] Cutout saved: %s", job_id, p.name)
        finally:
            try:
                from intake.detect import unload_model as _unload_seg
                _unload_seg()
            except Exception:
                pass

        if not all_cutouts:
            raise ValueError("No garment cutouts produced — check source photos.")

        logger.info("[%s] Phase 2 done — %d cutout(s)",
                    job_id, len(all_cutouts))

        # ── Phase 3: embed + tag + store (CLIP + Ollama) ───────────────
        items_added = 0
        try:
            from intake.embed import embed_image, unload_model as _unload_clip
            from intake.tag   import classify_vibes, dominant_color_name
            from db.chroma    import add_item
            from db.sqlite    import get_connection

            for label, cutout_path in all_cutouts:
                try:
                    item_id   = f"item_{uuid.uuid4().hex[:8]}"
                    embedding = embed_image(cutout_path)
                    vibes     = classify_vibes(embedding, top_k=3)
                    color     = dominant_color_name(cutout_path)
                    category  = _label_to_category(label)

                    metadata = {
                        "category":       category,
                        "dominant_color": color,
                        "vibe_tags":      json.dumps(vibes),
                        "image_path":     str(cutout_path),
                    }
                    add_item(item_id, embedding, metadata)

                    with get_connection() as conn:
                        conn.execute(
                            """INSERT OR REPLACE INTO wardrobe_items
                               (id, image_path, category, dominant_color, vibe_tags)
                               VALUES (?, ?, ?, ?, ?)""",
                            (item_id, str(cutout_path), category, color, json.dumps(vibes)),
                        )

                    items_added += 1
                    _set(job_id, items_added=items_added)
                    logger.info("[%s] ✓ %s — %s / %s / %s",
                                job_id, item_id, category, color, vibes)

                except Exception as exc:
                    logger.error("[%s] Failed to embed %s: %s",
                                 job_id, cutout_path.name, exc)
        finally:
            try:
                from intake.embed import unload_model as _unload_clip
                _unload_clip()
            except Exception:
                pass

        logger.info("[%s] Phase 3 done — %d item(s) stored",
                    job_id, items_added)

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", job_id, exc, exc_info=True)
        _set(job_id, status="failed", error=str(exc))
        return

    finally:
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)

    _set(job_id, status="done")
    logger.info("[%s] ✅ Pipeline complete — %d item(s) added",
                job_id, _jobs[job_id]["items_added"])


_CATEGORY_MAP = {
    "top":       "top",
    "bottom":    "bottom",
    "dress":     "dress",
    "outerwear": "outerwear",
    "shoes":     "shoes",
    "bag":       "bag",
    "accessory": "accessory",
    "hat":       "accessory",
    "scarf":     "accessory",
}


def _label_to_category(label: str) -> str:
    return _CATEGORY_MAP.get(label.lower(), label.lower())


def _set(job_id: str, **kwargs) -> None:
    if job_id not in _jobs:
        _jobs[job_id] = {"status": "queued", "items_added": 0, "total": 0, "error": None}
    _jobs[job_id].update(kwargs)
    if len(_jobs) > MAX_JOBS:
        evictable = [k for k, v in _jobs.items() if v["status"] in ("done", "failed")]
        for k in evictable[: len(_jobs) - MAX_JOBS]:
            del _jobs[k]
