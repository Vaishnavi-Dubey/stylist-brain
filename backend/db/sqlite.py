"""
sqlite.py — SQLite schema + connection helper
Stores outfit history for the habit lock system.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parents[2] / "stylist.db"


@contextmanager
def get_connection():
    """Context manager that auto-commits and closes the connection."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent readers + one writer without "database is locked"
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL, faster than FULL
    conn.execute("PRAGMA busy_timeout=5000")    # wait up to 5s before raising OperationalError
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't already exist. Safe to call repeatedly."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS outfit_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                combo_key   TEXT    NOT NULL,          -- sorted item IDs joined by |
                item_ids    TEXT    NOT NULL,          -- JSON array of item IDs
                unlocked    INTEGER NOT NULL DEFAULT 0,-- 1 = manually unlocked
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_combo_key ON outfit_history(combo_key);

            CREATE TABLE IF NOT EXISTS wardrobe_items (
                id             TEXT PRIMARY KEY,
                image_path     TEXT NOT NULL,
                category       TEXT,
                dominant_color TEXT,
                vibe_tags      TEXT,                  -- JSON array
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
    logger.info("SQLite DB ready at %s", DB_PATH)
