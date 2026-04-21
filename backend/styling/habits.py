"""
habits.py — Outfit history tracker + habit lock system
Stores every suggested outfit in SQLite and flags over-repeated combinations.
"""

import json
import logging

from db.sqlite import get_connection

logger = logging.getLogger(__name__)

LOCK_THRESHOLD = 3   # flag a combo as "locked" after this many appearances


def record_outfit(item_ids: list[str]) -> None:
    """
    Persist a suggested outfit to the history table.

    Args:
        item_ids: Ordered list of item IDs in the outfit.
    """
    combo_key = _make_key(item_ids)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO outfit_history (combo_key, item_ids) VALUES (?, ?)",
            (combo_key, json.dumps(sorted(item_ids))),
        )
    logger.info("Recorded outfit: %s", combo_key)


def get_locked_combos() -> list[list[str]]:
    """
    Return all item ID combinations suggested LOCK_THRESHOLD+ times that
    have not been manually unlocked.

    Counting is done entirely in SQLite (GROUP BY + HAVING) — no Python loop
    over the full history table.

    Returns:
        List of item ID lists to pass to the LLM as exclusions.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT item_ids
            FROM outfit_history
            WHERE unlocked = 0
            GROUP BY combo_key
            HAVING COUNT(*) >= ?
            """,
            (LOCK_THRESHOLD,),
        ).fetchall()

    return [json.loads(row["item_ids"]) for row in rows]


def unlock_combo(item_ids: list[str]) -> int:
    """
    Manually unlock a habit-locked outfit combination.

    Args:
        item_ids: The item IDs forming the combination to unlock.

    Returns:
        Number of history rows updated.
    """
    combo_key = _make_key(item_ids)
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE outfit_history SET unlocked = 1 WHERE combo_key = ?",
            (combo_key,),
        )
    logger.info("Unlocked combo: %s (%d rows)", combo_key, cursor.rowcount)
    return cursor.rowcount


def _make_key(item_ids: list[str]) -> str:
    """Stable, order-independent key for a set of item IDs."""
    return "|".join(sorted(item_ids))
