"""
gap.py — Wardrobe gap analyser
Inspects Ollama's outfit output and surfaces missing-piece diagnostics.
"""

import logging

logger = logging.getLogger(__name__)


def extract_gap(outfit_json: dict) -> dict | None:
    """
    Parse the gap block from an Ollama outfit response.

    Returns the gap dict if a missing item was identified, else None.

    Example return:
        {
            "missing_item": "white leather belt",
            "diy_hack": "knot the hem of your white shirt at the waist"
        }
    """
    gap = outfit_json.get("gap") or {}
    if not isinstance(gap, dict):
        return None
    missing = (gap.get("missing_item") or "").strip()

    if not missing or missing.lower() in ("none", "n/a", ""):
        return None

    return {
        "missing_item": missing,
        "diy_hack": gap.get("diy_hack") or "No hack available.",
    }


def has_complete_outfit(outfit_json: dict) -> bool:
    """
    Validate that the outfit contains exactly 3 items (Rule of Three).

    Args:
        outfit_json: Parsed Ollama response dict.

    Returns:
        True if the outfit field contains exactly 3 non-empty item IDs.
    """
    items = outfit_json.get("outfit", [])
    valid = [i for i in items if i and str(i).strip()]
    complete = len(valid) == 3
    if not complete:
        logger.warning("Incomplete outfit: got %d item(s), expected 3", len(valid))
    return complete
