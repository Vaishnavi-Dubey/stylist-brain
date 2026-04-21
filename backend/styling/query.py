"""
query.py — Vibe query engine
Converts a natural-language style query into a ranked list of wardrobe items
via CLIP text embedding + ChromaDB cosine similarity search.

RAM note: embed_text() loads CLIP (~340 MB). The route handler calls
unload_model() immediately after this function returns, so CLIP is free
before Ollama is invoked (~5.5 GB). Never overlap the two.
"""

import json
import logging
from typing import Any

from db.chroma import get_collection

logger = logging.getLogger(__name__)

TOP_N_DEFAULT = 15   # retrieve more candidates so Ollama has good choices


def query_wardrobe(
    vibe: str,
    top_n: int = TOP_N_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Embed *vibe* with CLIP text encoder and return top-N matching wardrobe items.

    Retrieves candidates from ALL categories so Ollama can assemble a
    complete Rule-of-Three outfit (top + bottom + third piece).

    Args:
        vibe:  Natural-language style query, e.g. "sharp Monday meeting".
        top_n: Number of candidates to retrieve (should be ≥ 9 to cover 3 categories).

    Returns:
        List of item dicts: { id, metadata { category, dominant_color, vibe_tags, image_path }, distance }
    """
    from intake.embed import embed_text   # imported lazily — CLIP loads here

    logger.info("Querying wardrobe for: %r (top_n=%d)", vibe, top_n)

    embedding  = embed_text(vibe)
    collection = get_collection()

    # Clamp top_n to collection size to avoid ChromaDB errors on small wardrobes
    count = collection.count()
    if count == 0:
        logger.warning("Wardrobe is empty — no items to query")
        return []
    n = min(top_n, count)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["metadatas", "distances"],
    )

    items: list[dict[str, Any]] = []
    for item_id, meta, dist in zip(
        results.get("ids", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        # vibe_tags is stored as a JSON string — deserialise for the LLM
        if isinstance(meta.get("vibe_tags"), str):
            try:
                meta = dict(meta)
                meta["vibe_tags"] = json.loads(meta["vibe_tags"])
            except (json.JSONDecodeError, TypeError):
                pass
        items.append({"id": item_id, "metadata": meta, "distance": round(dist, 4)})

    logger.info("Retrieved %d candidate(s) from ChromaDB", len(items))
    return items

