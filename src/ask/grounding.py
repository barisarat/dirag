"""Grounding / verification - the two judge loops of the answer pipeline.

Each judge is passed the resolved LLMConfig, so it answers with the corpus's
configured model and never picks a provider itself.

  - grade_sufficiency: does the retrieved context support answering the question?
    Used INSIDE retrieval to re-retrieve once when context is thin.
  - verify_grounding: does the drafted answer hold up against its cited passages?
    Used AFTER drafting to flag/revise unsupported claims.

Both are bounded (one call each) and fail OPEN: if the judge errors or returns
junk, we proceed rather than block the answer.
"""

import logging

from .json_utils import extract_json
from .llm import chat_create, text_of

logger = logging.getLogger(__name__)


def _parse_json(text):
    """First JSON object from a model response (tolerates fences/prose); {} on failure."""
    return extract_json(text, {})


def grade_sufficiency(cfg, query, context_blocks):
    """Judge whether the retrieved passages can answer the query.

    Returns {"sufficient": bool, "reformulation": str|None, "ok": bool}. Fails
    open (sufficient=True) so a flaky judge never blocks a usable answer.
    """
    if not context_blocks:
        return {"sufficient": False, "reformulation": None, "ok": True}

    prompt = (
        "You grade retrieval quality for a document Q&A system. Given the user's "
        "question and the retrieved passages, decide if the passages contain enough to "
        "answer it. If not, propose ONE better search query (different keywords/synonyms) "
        "that would likely retrieve the missing material.\n\n"
        'Respond with ONLY JSON: {"sufficient": true|false, "reformulation": "<query or null>"}\n\n'
        f"Question: {query}\n\nRetrieved passages:\n{context_blocks}"
    )

    try:
        response = chat_create(
            cfg,
            messages=[{"role": "user", "content": prompt}],
            # Headroom so a cloud reasoning model does not spend the whole cap on
            # hidden reasoning and truncate the tiny JSON verdict to empty.
            max_tokens=1000,
            temperature=0,  # deterministic JSON verdict (ignored on cloud reasoning models)
        )
        data = _parse_json(text_of(response))
        reformulation = data.get("reformulation")
        if isinstance(reformulation, str):
            reformulation = reformulation.strip() or None
        else:
            reformulation = None
        # A weak judge often echoes the original question as its "reformulation"; re-running
        # the same query just re-retrieves the same passages (a no-op after dedup), so drop it.
        if reformulation and reformulation.lower() == query.strip().lower():
            reformulation = None
        return {
            "sufficient": bool(data.get("sufficient", True)),
            "reformulation": reformulation,
            "ok": True,
        }
    except Exception:
        logger.debug("grade_sufficiency failed; treating context as sufficient", exc_info=True)
        return {"sufficient": True, "reformulation": None, "ok": False}


def verify_grounding(cfg, answer, context_blocks):
    """Check the drafted answer's claims against the cited passages.

    Returns {"grounded": bool, "unsupported": [str], "revised_answer": str|None,
    "ok": bool}. revised_answer (when present) drops/softens claims not supported
    by the sources. Fails open (grounded=True, no revision) on any error.
    """
    if not answer or not context_blocks:
        return {"grounded": True, "unsupported": [], "revised_answer": None, "ok": True}

    prompt = (
        "You are a strict fact-checker for a document assistant. Check the draft "
        "answer ONLY against the numbered sources. Every factual claim must be directly "
        "supported by a source. List any claims that are not supported. If there are "
        "unsupported claims, rewrite the answer to remove or appropriately hedge them "
        "while keeping the inline [n] citations and page/line references intact; otherwise "
        "return the answer unchanged.\n\n"
        'Respond with ONLY JSON: {"grounded": true|false, '
        '"unsupported": ["<claim>", ...], "revised_answer": "<full answer text>"}\n\n'
        f"Sources:\n{context_blocks}\n\nDraft answer:\n{answer}"
    )

    try:
        response = chat_create(
            cfg,
            messages=[{"role": "user", "content": prompt}],
            # Room for the full revised answer plus a cloud model's hidden reasoning.
            max_tokens=1500,
            temperature=0,  # deterministic JSON verdict (ignored on cloud reasoning models)
        )
        data = _parse_json(text_of(response))
        grounded = bool(data.get("grounded", True))
        unsupported = data.get("unsupported") or []
        if not isinstance(unsupported, list):
            unsupported = []
        unsupported = [str(u) for u in unsupported][:10]
        revised = data.get("revised_answer")
        revised = revised.strip() if isinstance(revised, str) and revised.strip() else None
        # A weak local judge frequently self-contradicts: grounded=true yet still lists
        # unsupported claims (and hedges them in revised_answer). Treat the draft as
        # ungrounded whenever any claim is flagged, and always pass the rewrite through - the
        # caller's accept_revision guard decides whether to actually swap it in.
        return {
            "grounded": grounded and not unsupported,
            "unsupported": unsupported,
            "revised_answer": revised,
            "ok": True,
        }
    except Exception:
        logger.debug("verify_grounding failed; treating answer as grounded", exc_info=True)
        return {"grounded": True, "unsupported": [], "revised_answer": None, "ok": False}
