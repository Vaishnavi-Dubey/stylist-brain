"""
chroma.py — ChromaDB client + wardrobe collection helpers
Persists garment embeddings to disk — no RAM spike from full collection loads.

ChromaDB automatically persists to ./chroma_store/ on every write.
"""

import logging
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = Path(__file__).parents[2] / "chroma_store"
COLLECTION_NAME    = "wardrobe"


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def init_chroma() -> None:
    """Called on app startup to warm up the ChromaDB client."""
    client = _get_client()
    client.get_or_create_collection(COLLECTION_NAME)
    logger.info("ChromaDB ready at %s", CHROMA_PERSIST_DIR)


def get_collection():
    """Return the wardrobe collection (creates it if missing)."""
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine distance
    )


def add_item(
    item_id: str,
    embedding: list[float],
    metadata: dict,
) -> None:
    """
    Add or update a wardrobe item in ChromaDB.

    Args:
        item_id:   Unique string ID for this garment.
        embedding: 512-d CLIP embedding list.
        metadata:  Dict with keys like category, dominant_color, vibe_tags.
    """
    collection = get_collection()
    collection.upsert(
        ids=[item_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    logger.debug("Upserted item %s to ChromaDB", item_id)


def delete_item(item_id: str) -> None:
    """Remove a wardrobe item from ChromaDB."""
    get_collection().delete(ids=[item_id])
    logger.info("Deleted item %s from ChromaDB", item_id)


def list_items(limit: int = 100) -> list[dict]:
    """
    Return all stored wardrobe items (id + metadata).

    Note: ChromaDB loads lazily — this does NOT pull all embeddings into RAM.
    """
    collection = get_collection()
    result = collection.get(limit=limit, include=["metadatas"])
    return [
        {"id": id_, "metadata": meta}
        for id_, meta in zip(result["ids"], result["metadatas"])
    ]
