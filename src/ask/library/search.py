"""Corpus retrieval: embed the question, KNN over the store, shape for answer.py.

Turns a question into the shared source dicts (url/title/text/score/chunkId/
metadata) that answer.py ranks and cites. Retrieval is the only corpus-specific
step; everything downstream (ranking, prompting, grounding, citations) is
shared.
"""

import json
from pathlib import Path

from .embedder import embed_texts


def _book_label(path):
    """Human label for citations: the book's file name without the .pdf suffix."""
    return Path(path).stem


def _to_source(row):
    try:
        bbox = json.loads(row["bbox_json"]) if row.get("bbox_json") else None
    except (ValueError, TypeError):
        bbox = None
    # Cosine distance in [0, 2]; convert to a higher-is-better score for answer.py.
    score = 1.0 - float(row.get("distance", 1.0))
    path = row["path"]
    return {
        "url": path,                       # grouping ref (per-book)
        "title": _book_label(path),
        "text": row.get("text", ""),
        "score": score,
        "chunkId": str(row.get("chunk_id", f"{path}:{row.get('page')}")),
        "sourceType": "library",
        "metadata": {"page": row.get("page"), "bbox": bbox},
    }


def search(store, question, k=8, embed_fn=embed_texts):
    """Retrieve up to k nearest chunks for a question as shared source dicts."""
    question = (question or "").strip()
    if not question:
        return []
    qvec = embed_fn([question])
    if not qvec:
        return []
    rows = store.search(qvec[0], k=k)
    return [_to_source(r) for r in rows]
