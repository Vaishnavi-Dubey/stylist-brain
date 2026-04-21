"""
routes.py — Styling engine FastAPI endpoints

POST  /styling/outfit              — vibe query → full outfit (CLIP → Ollama → flatlay)
GET   /styling/history             — recent outfit suggestions
GET   /styling/locked              — habit-locked combinations
POST  /styling/unlock              — manually unlock a habit combo
POST  /styling/illustrate          — trigger SD illustration as background job
GET   /styling/illustrate/{gen_id} — poll SD generation status
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from context.weather import get_weather
from context.calendar import get_todays_event
from db.sqlite import get_connection
from styling.habits import get_locked_combos, record_outfit, unlock_combo
from styling.gap import extract_gap
from styling.llm import build_outfit_prompt, call_ollama, SYSTEM_PROMPTS
from styling.query import query_wardrobe

logger = logging.getLogger(__name__)
router = APIRouter()

_OUTPUT_URL_PREFIX = "/output"   # matches StaticFiles mount in main.py


# ── Request / Response models ────────────────────────────────────────────────────

class OutfitRequest(BaseModel):
    vibe:  str  = Field(..., min_length=3, max_length=500,
                        example="sharp but approachable for a rainy Monday meeting")
    city:  str  = Field(default="", description="Override weather city (optional)")
    model: str  = Field(default="llama3.2:3b",
                        description="Ollama model tag — use llama3:8b or mistral:7b only")


class IllustrateRequest(BaseModel):
    outfit_json:   dict            = Field(..., description="Full outfit JSON from /styling/outfit")
    item_metadata: list[dict]      = Field(default=[], description="Item metadata list (optional)")
    steps:         int             = Field(default=20, ge=1, le=25,
                                          description="SD inference steps — max 25 on MacBook Air")


class UnlockRequest(BaseModel):
    item_ids: list[str] = Field(..., min_length=1,
                                description="Item IDs of the combination to unlock")


# ── Routes ───────────────────────────────────────────────────────────────────────

@router.post("/outfit", summary="Generate 3 outfit suggestions from a vibe query")
def get_outfit(req: OutfitRequest) -> dict[str, Any]:
    """
    Full styling pipeline returning 3 outfit suggestions at different creativity levels:
      safe (temp 0.7) · creative (temp 1.0) · experimental (temp 1.2)

    Each suggestion includes a flat-lay composite image.
    """
    # ── 1. Context ──────────────────────────────────────────────────────────────
    city    = req.city.strip() or None
    weather = get_weather(city) if city else get_weather()
    event   = get_todays_event()
    logger.info("Weather: %s | Event: %s", weather, event)

    # ── 2. Habit locks ──────────────────────────────────────────────────────────
    locked = get_locked_combos()

    # ── 3. CLIP retrieval ───────────────────────────────────────────────────────
    try:
        items = query_wardrobe(req.vibe, top_n=15)
    finally:
        try:
            from intake.embed import unload_model as unload_clip
            unload_clip()
        except Exception:
            pass

    if not items:
        raise HTTPException(
            status_code=404,
            detail="Your wardrobe is empty. Upload some items first via POST /intake/upload.",
        )

    if len(items) < 3:
        logger.warning("Only %d item(s) in wardrobe — outfit may be incomplete", len(items))

    # ── 4. Build shared prompt ──────────────────────────────────────────────────
    prompt = build_outfit_prompt(
        vibe=req.vibe,
        items=items,
        weather=weather,
        calendar_event=event,
        locked_combos=locked if locked else None,
    )

    valid_ids = {item["id"] for item in items}

    # ── 5. Three Ollama calls with temperature variation ────────────────────────
    _PERSONA_CONFIGS = [
        ("safe",         0.7,  SYSTEM_PROMPTS["safe"]),
        ("creative",     1.0,  SYSTEM_PROMPTS["creative"]),
        ("experimental", 1.2,  SYSTEM_PROMPTS["experimental"]),
    ]

    suggestions: list[dict] = []
    for i, (style_name, temp, sys_prompt) in enumerate(_PERSONA_CONFIGS, start=1):
        try:
            outfit_json = call_ollama(
                prompt,
                model=req.model,
                temperature=temp,
                system_prompt=sys_prompt,
            )
        except RuntimeError as exc:
            logger.warning("Suggestion %d (%s) failed: %s", i, style_name, exc)
            continue

        # Validate returned item IDs (hallucination guard)
        raw_outfit   = outfit_json.get("outfit", [])
        clean_outfit = [oid for oid in raw_outfit if oid in valid_ids]
        if len(clean_outfit) < len(raw_outfit):
            logger.warning(
                "Suggestion %d hallucinated IDs: %s", i,
                set(raw_outfit) - valid_ids,
            )
        outfit_json["outfit"] = clean_outfit

        # Gap analysis
        gap = extract_gap(outfit_json)
        outfit_json["gap"] = gap or {"missing_item": None, "diy_hack": None}

        # Record in history
        if clean_outfit:
            record_outfit(clean_outfit)

        # Flat-lay for this suggestion
        flatlay_url: str | None = None
        outfit_meta = [
            next((it for it in items if it["id"] == oid), {})
            for oid in clean_outfit
        ]
        try:
            from imaging.flatlay import create_flatlay
            technique    = (outfit_json.get("styling_techniques") or [""])[0]
            flatlay_path = create_flatlay(
                item_ids=clean_outfit,
                technique=technique,
                item_metadata=outfit_meta,
                filename=f"outfit_{i}.png",
            )
            flatlay_url = f"{_OUTPUT_URL_PREFIX}/{Path(flatlay_path).name}"
        except Exception as exc:
            logger.warning("Flat-lay for suggestion %d failed: %s", i, exc)

        outfit_json["images"] = {"flatlay": flatlay_url, "generated": None}
        outfit_json["_style"] = style_name   # hint for frontend tab label
        suggestions.append(outfit_json)

    if not suggestions:
        raise HTTPException(status_code=503, detail="Ollama failed to return any valid outfit suggestions.")

    return {"suggestions": suggestions}


@router.get("/history", summary="Recent outfit suggestions")
def get_history(limit: int = 20) -> dict[str, Any]:
    """
    Return the most recent outfit suggestions with per-item metadata
    (image URL, category, dominant_color) so the frontend can show thumbnails.
    """
    import json as _json
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, combo_key, item_ids, unlocked, created_at
               FROM outfit_history
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    history = []
    with get_connection() as conn:
        for row in rows:
            try:
                ids = _json.loads(row["item_ids"])
            except Exception:
                ids = row["combo_key"].split("|")

            # Fetch metadata for each item in the outfit
            items_meta = []
            for item_id in ids:
                meta_row = conn.execute(
                    "SELECT category, dominant_color FROM wardrobe_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
                items_meta.append({
                    "id":            item_id,
                    "image_url":     f"/intake/image/{item_id}",
                    "category":      meta_row["category"]       if meta_row else None,
                    "dominant_color": meta_row["dominant_color"] if meta_row else None,
                })

            history.append({
                "id":         row["id"],
                "items":      items_meta,
                "unlocked":   row["unlocked"],
                "created_at": row["created_at"],
            })

    return {"count": len(history), "history": history}


@router.get("/locked", summary="View habit-locked outfit combinations")
def get_locked() -> dict[str, Any]:
    """
    Returns all combinations suggested 3+ times that haven't been manually unlocked.
    These are excluded from future suggestions automatically.
    """
    locked = get_locked_combos()
    return {
        "count":              len(locked),
        "locked_combinations": locked,
        "note": f"Combinations suggested {3}+ times are excluded from future outfits.",
    }


@router.post("/unlock", summary="Unlock a habit-locked outfit combination")
def unlock(req: UnlockRequest) -> dict[str, Any]:
    """
    Manually unlock a habit-locked combination so it can be suggested again.

    Send the item IDs of the combination you want to re-enable.
    """
    rows = unlock_combo(req.item_ids)
    if rows == 0:
        raise HTTPException(
            status_code=404,
            detail="No locked combination found for those item IDs.",
        )
    return {"unlocked": True, "item_ids": req.item_ids, "rows_affected": rows}


# ── Mode 2: SD Illustration ────────────────────────────────────────────────────

@router.post("/illustrate", summary="Start a Stable Diffusion illustration job")
def start_illustrate(req: IllustrateRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Kick off an AI illustration generation job as a background task.

    SD inference takes 20–40 s on MacBook Air — this endpoint returns
    immediately with a ``gen_id``.  Poll ``GET /styling/illustrate/{gen_id}``
    until ``status == "done"``.

    RAM note: SD pipeline (~2–4 GB) must NOT overlap with CLIP/YOLO.
    If you're mid-intake, wait for the job to finish before calling this.
    """
    from imaging.generate import (
        _run_generation_job,
        start_generation_job,
    )

    gen_id = start_generation_job(
        outfit_json=req.outfit_json,
        item_metadata=req.item_metadata or None,
        steps=req.steps,
    )

    background_tasks.add_task(
        _run_generation_job,
        gen_id,
        req.outfit_json,
        req.item_metadata or None,
        req.steps,
    )

    logger.info("SD illustration job %s queued (%d steps)", gen_id, req.steps)
    return {
        "gen_id":  gen_id,
        "status":  "queued",
        "message": "Generation started. Poll GET /styling/illustrate/{gen_id} for status.",
    }


@router.get("/illustrate/{gen_id}", summary="Poll an SD illustration job")
def poll_illustrate(gen_id: str) -> dict[str, Any]:
    """
    Returns the current status of an SD generation job.

    Possible statuses: ``queued`` → ``running`` → ``done`` | ``failed``

    When ``status == "done"``, the ``image_url`` field contains the
    ``/output/…`` URL ready to display in the frontend.
    """
    from imaging.generate import get_generation_status

    job = get_generation_status(gen_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Generation job '{gen_id}' not found.")

    image_url: str | None = None
    if job["status"] == "done" and job.get("path"):
        image_url = f"{_OUTPUT_URL_PREFIX}/{Path(job['path']).name}"

    return {
        "gen_id":    gen_id,
        "status":    job["status"],
        "image_url": image_url,
        "error":     job.get("error"),
    }


@router.get("/sd-status", summary="Check whether SD pipeline is loaded in memory")
def sd_status() -> dict[str, Any]:
    """Quick health check — tells the frontend if SD is already warm."""
    try:
        from imaging.generate import is_sd_loaded
        loaded = is_sd_loaded()
    except Exception:
        loaded = False
    return {"sd_loaded": loaded}
