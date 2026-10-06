"""Rerankers: reorder the top of a result list by reading query and passage together.

    neural   a cross-encoder (fastembed TextCrossEncoder), local. Scores the top
             NEURAL_DEPTH passages.
             DIRAG_RERANK_MODEL   default Xenova/ms-marco-MiniLM-L-12-v2; any model
                                  fastembed lists as a cross-encoder works, e.g.
                                  BAAI/bge-reranker-base (larger, slower)
             It runs where the embedder runs (see embed.py).
    llm      the language model through Ollama (see llm.py).
             Orders the top LLM_DEPTH passages and leaves out those that do not
             help answer the query; those stay in the list, after the kept ones,
             marked `dropped`.

Passages are scored with their book and chapter in front, as they were embedded.
"""

import os
import threading

from . import config, embed, llm

NEURAL_MODEL = os.getenv("DIRAG_RERANK_MODEL") or "Xenova/ms-marco-MiniLM-L-12-v2"
NEURAL_DEPTH = 30
LLM_DEPTH = 20
LLM_CHARS = 700

_cross = None
_lock = threading.Lock()


def _document(row):
    return f"{row['book']} > {row['section'] or ''}\n\n{row['text']}"


def neural(question, rows):
    """`rows` reordered: the top NEURAL_DEPTH by cross-encoder score, the rest unchanged."""
    global _cross
    with _lock:
        if _cross is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _cross = embed.build(lambda **o: TextCrossEncoder(model_name=NEURAL_MODEL, cache_dir=str(config.MODELS), **o),
                                 lambda m: list(m.rerank("probe", ["probe"])))
    top, rest = rows[:NEURAL_DEPTH], rows[NEURAL_DEPTH:]
    scores = list(_cross.rerank(question, [_document(row) for row in top]))
    return [row for _, row in sorted(zip(scores, top), key=lambda pair: -pair[0])] + rest


LLM_SYSTEM = ("You rank book passages for a search query. Read every passage, then reply with JSON only: "
              '{"ranking": [passage numbers]}, most useful first. Leave out passages that do not help answer '
              "the query. An empty ranking is a valid answer.")


def by_llm(question, rows):
    """`rows` reordered by the language model: kept passages first, then the dropped ones, then the rest."""
    top, rest = rows[:LLM_DEPTH], rows[LLM_DEPTH:]
    listing = "\n\n".join(f"[{n}] {row['book']} > {row['section'] or ''}, p. {row['page']}\n{row['text'][:LLM_CHARS]}"
                          for n, row in enumerate(top, 1))
    reply = llm.chat_json(LLM_SYSTEM, f"Query: {question}\n\nPassages:\n\n{listing}")
    order = []
    for value in reply.get("ranking", []) if isinstance(reply, dict) else []:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(top) and n - 1 not in order:
            order.append(n - 1)
    kept = [top[i] for i in order]
    dropped = [dict(row, dropped=True) for i, row in enumerate(top) if i not in order]
    return kept + dropped + rest
