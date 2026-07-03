"""Library search: KNN rows mapped to the shared source shape (no LLM, no network)."""

from ask.library import index as index_mod, search as lib_search
from ask.library.embedder import EMBED_DIM
from ask.library.store import Store


def _distinct_embed(texts):
    """Deterministic but content-varying embeddings so KNN ordering is meaningful."""
    out = []
    for t in texts:
        seed = (len(t) % 7) + 1
        out.append([float(seed)] * EMBED_DIM)
    return out


def test_search_returns_shared_source_shape(tmp_path, library_dir):
    store = Store(tmp_path / "lib.sqlite3")
    index_mod.build_index(store, library_dir, embed_fn=_distinct_embed, use_ocr=False)

    sources = lib_search.search(store, "how are backups pruned", k=5, embed_fn=_distinct_embed)
    assert sources, "expected at least one hit"

    s = sources[0]
    # Exactly the keys answer.py consumes.
    for key in ("url", "title", "text", "score", "chunkId", "metadata"):
        assert key in s
    assert s["sourceType"] == "library"
    assert s["metadata"].get("page") is not None
    assert s["title"] in {"backups", "servers"}  # file stem, no .pdf
    assert isinstance(s["score"], float)


def test_search_empty_question_returns_nothing(tmp_path, library_dir):
    store = Store(tmp_path / "lib.sqlite3")
    index_mod.build_index(store, library_dir, embed_fn=_distinct_embed, use_ocr=False)
    assert lib_search.search(store, "   ", k=5, embed_fn=_distinct_embed) == []


def test_to_source_maps_bbox_and_score():
    row = {
        "chunk_id": 42,
        "path": "/books/networking.pdf",
        "page": 12,
        "bbox_json": "[1.0, 2.0, 3.0, 4.0]",
        "text": "TCP provides reliable delivery.",
        "distance": 0.25,
    }
    s = lib_search._to_source(row)
    assert s["title"] == "networking"
    assert s["metadata"]["page"] == 12
    assert s["metadata"]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert abs(s["score"] - 0.75) < 1e-9  # 1 - distance
    assert s["chunkId"] == "42"


def test_to_source_tolerates_bad_bbox():
    row = {"chunk_id": 1, "path": "/b.pdf", "page": 1, "bbox_json": "not json", "text": "x", "distance": 0.0}
    s = lib_search._to_source(row)
    assert s["metadata"]["bbox"] is None
