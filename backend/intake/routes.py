"""
routes.py — Wardrobe intake FastAPI endpoints

POST  /intake/upload            — upload video(s) or image(s), returns job_id
GET   /intake/status/{job_id}   — poll pipeline progress
GET   /intake/items             — list all catalogued wardrobe items
DELETE /intake/items/{item_id}  — remove a single item
"""

import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from db.chroma import list_items, delete_item as chroma_delete, get_collection
from db.sqlite import get_connection
from intake.pipeline import run_pipeline, _jobs

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path(__file__).parents[2] / "wardrobe" / "uploads"

_ACCEPTED_MIME_PREFIXES = ("video/", "image/")


@router.post("/upload", summary="Upload wardrobe video or item photos")
async def upload_wardrobe(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Video pan or individual item photos"),
):
    """
    Accept one or more files and queue the full intake pipeline as a background job.

    - Videos: extracted → detected → segmented → embedded → stored
    - Images: detected → segmented → embedded → stored

    Returns a `job_id` you can poll with `GET /intake/status/{job_id}`.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    # Validate MIME types before touching disk
    for f in files:
        ct = f.content_type or ""
        if not any(ct.startswith(p) for p in _ACCEPTED_MIME_PREFIXES):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ct}'. Send video/* or image/*.",
            )

    job_id = uuid.uuid4().hex[:8]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for f in files:
        # Sanitise filename — keep only the last path component
        safe_name = Path(f.filename or "upload").name
        dest = job_upload_dir / safe_name

        # Stream in 1 MB chunks — never load an entire video into RAM at once
        with open(dest, "wb") as fh:
            while True:
                chunk = await f.read(1024 * 1024)   # 1 MB
                if not chunk:
                    break
                fh.write(chunk)

        saved_paths.append(dest)
        logger.info("[%s] Saved upload: %s (%d bytes)", job_id, safe_name, dest.stat().st_size)

    # Register job and kick off background processing
    _jobs[job_id] = {"status": "queued", "items_added": 0, "total": len(saved_paths), "error": None}
    background_tasks.add_task(run_pipeline, job_id, saved_paths)

    return {
        "job_id":      job_id,
        "files":       len(saved_paths),
        "status":      "queued",
        "poll_url":    f"/intake/status/{job_id}",
    }


@router.get("/status/{job_id}", summary="Poll intake job progress")
def get_status(job_id: str):
    """
    Returns the current state of an intake job.

    Status values:
    - `queued`     — waiting to start
    - `processing` — pipeline is running (check `items_added` for progress)
    - `done`       — finished successfully
    - `failed`     — pipeline error (see `error` field)
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _jobs[job_id]


@router.get("/items", summary="List all wardrobe items")
def list_wardrobe():
    """
    Return all catalogued wardrobe items with their metadata.
    Reads from SQLite (source of truth) — always reflects deletes immediately.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, image_path, category, dominant_color, vibe_tags
               FROM wardrobe_items ORDER BY created_at DESC"""
        ).fetchall()

    items = [
        {
            "id": row["id"],
            "metadata": {
                "image_path":     row["image_path"],
                "category":       row["category"],
                "dominant_color": row["dominant_color"],
                "vibe_tags":      row["vibe_tags"],
            },
        }
        for row in rows
    ]
    return {"count": len(items), "items": items}


@router.delete("/items", summary="Clear entire wardrobe")
def clear_wardrobe():
    """
    Delete every item from ChromaDB and SQLite in one shot.
    Drops and recreates the ChromaDB collection for a clean slate.
    """
    from db.chroma import _get_client
    client = _get_client()
    try:
        client.delete_collection("wardrobe")
    except Exception:
        pass
    client.get_or_create_collection("wardrobe", metadata={"hnsw:space": "cosine"})

    with get_connection() as conn:
        deleted = conn.execute("DELETE FROM wardrobe_items").rowcount

    logger.info("Cleared wardrobe — %d item(s) removed", deleted)
    return {"cleared": True, "items_removed": deleted}


@router.get("/image/{item_id}", summary="Serve a wardrobe item's image")
def get_item_image(item_id: str):
    """
    Return the raw image file for a wardrobe item.
    Looks up the filesystem path from SQLite then streams the file.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT image_path FROM wardrobe_items WHERE id = ?", (item_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    path = Path(row["image_path"])
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image file missing for '{item_id}'. Re-upload the item to restore it.",
        )

    # Detect media type from extension
    suffix = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    return FileResponse(str(path), media_type=media_type, headers={
        "Cache-Control": "public, max-age=86400",  # cache images for 1 day
    })


@router.delete("/items/{item_id}", summary="Remove a wardrobe item")
def remove_item(item_id: str):
    """
    Delete a wardrobe item from both ChromaDB and SQLite.
    The segment image on disk is NOT deleted (user may want to re-add it).
    """
    try:
        chroma_delete(item_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Item not found in vector DB: {exc}")

    with get_connection() as conn:
        conn.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))

    logger.info("Deleted wardrobe item: %s", item_id)
    return {"deleted": item_id}


@router.post("/reprocess-wardrobe", summary="Re-extract colors for all wardrobe items")
async def reprocess_wardrobe():
    """
    Re-run color extraction (K-means) on all existing segment images.
    Runs in a thread pool — returns immediately, logs progress to console.
    """
    import asyncio
    import json as _json
    from intake.tag import dominant_color_name
    from db.chroma import get_collection

    def _do_reprocess():
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, image_path FROM wardrobe_items"
            ).fetchall()

        processed = 0
        failed    = 0
        collection = get_collection()

        for row in rows:
            item_id    = row["id"]
            image_path = Path(row["image_path"]) if row["image_path"] else None
            if not image_path or not image_path.exists():
                failed += 1
                continue
            try:
                new_color = dominant_color_name(image_path)
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE wardrobe_items SET dominant_color = ? WHERE id = ?",
                        (new_color, item_id),
                    )
                try:
                    existing = collection.get(ids=[item_id], include=["metadatas"])
                    meta = (existing["metadatas"] or [{}])[0] or {}
                    meta["dominant_color"] = new_color
                    collection.update(ids=[item_id], metadatas=[meta])
                except Exception:
                    pass
                processed += 1
                logger.info("Reprocessed %s → %s", item_id, new_color)
            except Exception as exc:
                logger.error("Reprocess failed for %s: %s", item_id, exc)
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(rows)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_reprocess)
    return result
