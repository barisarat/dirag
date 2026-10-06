"""Retrieval over the index, then an optional rerank, then the per-book cap.

Modes:

    lexical    BM25 over passage text: exact terms.
    semantic   vector search over passage embeddings: meaning.
    hybrid     both, combined with reciprocal rank fusion (a passage ranked r
               adds 1/(RRF_K + r)); ranks are fused because BM25 and cosine
               distance are on different scales.

Type-ahead (`live`) is BM25 with the last word as a prefix, and no rerank.

Rerank (see rerank.py): off, neural (cross-encoder) or llm.

The query is used verbatim, never expanded or rewritten. Results are
deduplicated and capped per book, after any rerank, so sibling editions or
volumes of one series do not fill the list with the same paragraph.
"""

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec

from . import embed

MODES = ("lexical", "semantic", "hybrid")
RERANKS = ("off", "neural", "llm")

# Candidates each retriever contributes before fusion.
POOL = 120
RRF_K = 60


def connect(db_path):
    """Read-only connection with sqlite-vec loaded."""
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"No index at {path}. Build it with: dirag index")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def terms_of(text):
    """Words of a query; each is quoted later, so punctuation never reaches FTS5 as syntax."""
    return [t for t in re.findall(r"[\w][\w'-]*", text) if len(t) > 1]


def lexical(conn, terms, limit, join=" AND ", prefix=False):
    """BM25 candidates. With prefix=True the last term is a prefix match, for type-ahead."""
    if not terms:
        return []
    parts = [f'"{term}"' for term in terms]
    if prefix:
        parts[-1] += "*"
    try:
        rows = conn.execute("SELECT rowid AS chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?"
                            " ORDER BY bm25(chunks_fts) LIMIT ?", (join.join(parts), limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [row["chunk_id"] for row in rows]


def lexical_any(conn, terms, limit, prefix=False):
    """All terms first; any term when that returns too little, as long questions do."""
    ids = lexical(conn, terms, limit, prefix=prefix)
    return ids if len(ids) >= limit // 4 else lexical(conn, terms, limit, join=" OR ", prefix=prefix)


def vector(conn, question, limit):
    rows = conn.execute("SELECT chunk_id FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                        (json.dumps(embed.query(question)), limit)).fetchall()
    return [row["chunk_id"] for row in rows]


def fuse(*rankings):
    """Reciprocal rank fusion: [(chunk_id, score)], best first."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


def retrieve(conn, question, pool=POOL):
    """Hybrid candidates: BM25 and vectors, fused."""
    return fuse(vector(conn, question, pool), lexical_any(conn, terms_of(question), pool))


def retrieve_live(conn, question, pool=POOL):
    """Type-ahead candidates: BM25 only, last word as a prefix. No vector scan per keystroke."""
    return fuse(lexical_any(conn, terms_of(question), pool, prefix=True))


def hydrate(conn, chunk_ids):
    """Chunk rows joined to their book and section, keyed by chunk id."""
    if not chunk_ids:
        return {}
    marks = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        "SELECT c.id, c.page, c.text, c.bbox_json, b.id AS book_id, b.title AS book, b.author, b.year, b.path,"
        " s.path_text AS section, s.chapter, s.start_page, s.end_page"
        " FROM chunks c JOIN books b ON b.id = c.book_id LEFT JOIN sections s ON s.id = c.section_id"
        f" WHERE c.id IN ({marks})", chunk_ids).fetchall()
    return {row["id"]: dict(row) for row in rows}


def candidates(conn, question, mode="hybrid", live=False, pool=POOL):
    """Fused (chunk_id, score) candidates for a mode, best first."""
    if live:
        return retrieve_live(conn, question, pool)
    if mode == "lexical":
        return fuse(lexical_any(conn, terms_of(question), pool))
    if mode == "semantic":
        return fuse(vector(conn, question, pool))
    return retrieve(conn, question, pool)


def ordered(conn, ranked):
    """Rows for (chunk_id, score) pairs in order, without overlapping duplicates."""
    rows = hydrate(conn, [chunk_id for chunk_id, _ in ranked])
    seen, out = set(), []
    for chunk_id, _ in ranked:
        row = rows.get(chunk_id)
        # Windows overlap, so neighbouring passages can share a head.
        if row is None or row["text"][:80] in seen:
            continue
        seen.add(row["text"][:80])
        out.append(row)
    return out


def capped(rows, limit=20, per_book=3):
    """The first `limit` rows with at most `per_book` from any one book."""
    per, out = {}, []
    for row in rows:
        if per.get(row["path"], 0) >= per_book:
            continue
        per[row["path"]] = per.get(row["path"], 0) + 1
        out.append(row)
        if len(out) >= limit:
            break
    return out


def passages(conn, question, mode="hybrid", rerank="off", live=False, limit=20, per_book=3):
    """The result list: retrieve, rerank, cap. Raises llm.LLMError when an llm rerank fails."""
    from . import rerank as rerankers
    rows = ordered(conn, candidates(conn, question, mode, live))
    if not live and rerank == "neural":
        rows = rerankers.neural(question, rows)
    elif not live and rerank == "llm":
        rows = rerankers.by_llm(question, rows)
    return capped(rows, limit, per_book)


def chapters(conn, candidates, limit=10):
    """Passage hits rolled up to (book, chapter), ranked by summed fusion score."""
    rows = hydrate(conn, [chunk_id for chunk_id, _ in candidates])
    found = {}
    for chunk_id, score in candidates:
        row = rows.get(chunk_id)
        if row is None:
            continue
        entry = found.setdefault((row["path"], row["chapter"] or ""), {
            "book": row["book"], "year": row["year"], "chapter": row["chapter"] or "(no chapter map)",
            "score": 0.0, "hits": 0, "pages": []})
        entry["score"] += score
        entry["hits"] += 1
        entry["pages"].append(row["page"])
    return sorted(found.values(), key=lambda entry: -entry["score"])[:limit]
