"""Shared grounded-answer pipeline: context selection, prompt, citations.

Selection ranks retrieved chunks by grouping them per source, weighting query
coverage by term rarity, and keeping a secondary source only when it scores
within 82% of the best one. make_citation renders a citation surface of
book+page (with an optional line locator for line-addressed sources). The
answer is drafted, its citation markers validated, and a verification pass may
propose a rewrite that accept_revision only takes when it does not regress.

Retrieval feeds this one source shape - a list of dicts with keys:
  url      grouping ref (the source PDF's path)
  title    display label (the book title / file name)
  text     the retrieved passage
  score    retrieval score (cosine similarity)
  chunkId  stable id for dedup
  metadata {page, bbox}; {line} for line-addressed sources
"""

import re
from collections import defaultdict

from . import grounding, llm
from .text_utils import tokenize

# ----------------------------------------------------------------------------
# Ranking + citation core
# ----------------------------------------------------------------------------


def result_search_text(result):
    return "\n".join([
        result.get("title", ""),
        result.get("text", ""),
    ])


def _passage_snippet(text, limit=400):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def make_citation(result):
    meta = result.get("metadata") or {}
    return {
        # Display label: the book title / file name.
        "label": result.get("title", ""),
        # Grouping ref: the source PDF's path.
        "ref": result.get("url", ""),
        # Page locator (or a line locator for line-addressed sources), plus the
        # exact retrieved passage so the renderer can show where the answer came from.
        "page": meta.get("page"),
        "line": meta.get("line"),
        "passage": _passage_snippet(result.get("text", "")),
        # Bounding box [x0, y0, x1, y1] in PDF points; None for sources without
        # geometry (line-addressed text).
        "bbox": meta.get("bbox"),
    }


def group_results_by_url(results):
    groups = {}

    for result in results:
        url = result.get("url", "")

        if not url:
            continue

        if url not in groups:
            groups[url] = {
                "url": url,
                "results": [],
                "best_score": 0,
                "source_type": result.get("sourceType", ""),
            }

        groups[url]["results"].append(result)
        groups[url]["best_score"] = max(groups[url]["best_score"], float(result.get("score", 0)))

    for group in groups.values():
        group["results"].sort(key=lambda item: float(item.get("score", 0)), reverse=True)

    return list(groups.values())


def build_group_token_frequency(groups, query_tokens):
    frequency = defaultdict(int)

    for group in groups:
        group_text = " ".join(result_search_text(result) for result in group["results"])
        group_tokens = set(tokenize(group_text))

        for token in query_tokens:
            if token in group_tokens:
                frequency[token] += 1

    return frequency


def score_group(group, query_tokens, token_frequency, group_count):
    group_text = " ".join(result_search_text(result) for result in group["results"])
    group_tokens = set(tokenize(group_text))

    coverage_score = 0

    for token in query_tokens:
        if token not in group_tokens:
            continue

        frequency = token_frequency.get(token, group_count)
        rarity_weight = 1 / max(frequency, 1)
        coverage_score += rarity_weight

    normalized_coverage = coverage_score / max(len(query_tokens), 1)
    # Small nudge (<=0.045) toward sources corroborated by multiple chunks, without letting a
    # high-hit-count source dominate a single strongly-relevant one.
    multi_chunk_bonus = min(len(group["results"]), 3) * 0.015

    return float(group["best_score"]) + normalized_coverage + multi_chunk_bonus


def select_answer_context(query, results, max_items):
    if not results:
        return []

    query_tokens = list(dict.fromkeys(tokenize(query)))

    if not query_tokens:
        return results[:max_items]

    groups = group_results_by_url(results)

    if not groups:
        return results[:max_items]

    token_frequency = build_group_token_frequency(groups, query_tokens)

    for group in groups:
        group["group_score"] = score_group(
            group,
            query_tokens,
            token_frequency,
            len(groups),
        )

    groups.sort(key=lambda item: item["group_score"], reverse=True)

    best_group = groups[0]
    best_score = float(best_group["group_score"])
    selected = []

    for result in best_group["results"][:3]:
        selected.append(result)

        if len(selected) >= max_items:
            return selected

    for group in groups[1:]:
        group_score = float(group["group_score"])

        # Only pull in a secondary source if it scores within 82% of the best group - keeps
        # the answer focused on the strongest sources rather than padding with weak ones.
        if group_score < best_score * 0.82:
            continue

        selected.append(group["results"][0])

        if len(selected) >= max_items:
            return selected

    if len(selected) < max_items:
        seen = {
            result.get("chunkId", "")
            for result in selected
        }

        for result in results:
            chunk_id = result.get("chunkId", "")

            if chunk_id in seen:
                continue

            selected.append(result)
            seen.add(chunk_id)

            if len(selected) >= max_items:
                break

    return selected


def remove_invalid_citation_markers(answer, citation_count):
    def replace_marker(match):
        citation_number = int(match.group(1))

        if citation_number < 1 or citation_number > citation_count:
            return ""

        return match.group(0)

    cleaned = re.sub(r"\[(\d+)\]", replace_marker, answer)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def _citation_markers(text):
    return {int(n) for n in _CITATION_MARKER_RE.findall(text or "")}


def accept_revision(original, revised):
    """Accept a grounding judge's rewrite only if it doesn't regress the answer.

    The judge is the same model that drafted the answer, so its rewrite can drop
    [n] citations (defeating precise citations) or truncate. Keep the rewrite only
    when it preserves the original's citations (allowing at most one dropped marker
    for a legitimately removed unsupported claim) and isn't drastically shorter;
    otherwise the caller keeps the original.
    """
    if not revised or not revised.strip():
        return False
    orig = _citation_markers(original)
    new = _citation_markers(revised)
    if orig and len(orig & new) < max(1, len(orig) - 1):
        return False
    if len(revised.strip()) < 0.5 * len(original.strip()):
        return False
    return True


# ----------------------------------------------------------------------------
# Orchestration: the grounded-answer surface the CLI calls
# ----------------------------------------------------------------------------

ANSWER_INSTRUCTIONS = (
    "Answer the question using ONLY the numbered sources below. Cite every claim "
    "with its source marker, like [1] or [2]. Do not use any knowledge beyond the "
    "sources. If the sources do not contain the answer, say so plainly instead of "
    "guessing. Write plain prose only - no markdown, no headings, no bullet "
    "characters, no code fences."
)


def _locator(citation):
    if citation.get("page") is not None:
        return f"page {citation['page']}"
    if citation.get("line") is not None:
        return f"line {citation['line']}"
    return ""


def format_context(citations):
    """Numbered sources block fed to both the answer prompt and the grounding judge."""
    blocks = []
    for i, c in enumerate(citations, 1):
        loc = _locator(c)
        head = c.get("label", "") or c.get("ref", "")
        if loc:
            head = f"{head} ({loc})"
        blocks.append(f"[{i}] {head}\n{c.get('passage', '')}")
    return "\n\n".join(blocks)


def build_answer_prompt(question, context):
    return (
        f"{ANSWER_INSTRUCTIONS}\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(cfg, question, sources, max_items=6):
    """Draft a grounded, cited answer from retrieved sources, then verify it once.

    Returns a dict:
      answer      final answer text (revised only if the judge's rewrite is accepted)
      citations   list of make_citation dicts, index+1 == the [n] marker
      grounded    the verifier's verdict (fail-open True)
      revised     whether the grounding rewrite was accepted
      no_sources  True when retrieval returned nothing to answer from
    """
    selected = select_answer_context(question, sources, max_items)
    citations = [make_citation(s) for s in selected]

    if not citations:
        return {
            "answer": "",
            "citations": [],
            "grounded": True,
            "revised": False,
            "no_sources": True,
        }

    context = format_context(citations)
    prompt = build_answer_prompt(question, context)

    response = llm.chat_create(
        cfg,
        messages=[{"role": "user", "content": prompt}],
        # Headroom for cloud reasoning models (e.g. gpt-5-nano): hidden reasoning
        # tokens count against this cap, so too low truncates the answer to empty.
        max_tokens=1500,
        temperature=0,
    )
    draft = remove_invalid_citation_markers(llm.text_of(response), len(citations))

    verdict = grounding.verify_grounding(cfg, draft, context)
    final = draft
    revised_accepted = False
    revised = verdict.get("revised_answer")
    if revised:
        cleaned = remove_invalid_citation_markers(revised, len(citations))
        if accept_revision(draft, cleaned):
            final = cleaned
            revised_accepted = True

    # If we swapped in the judge's hedged rewrite, the flagged claims were addressed,
    # so the answer is grounded; only warn when an unaddressed flag survives.
    grounded = bool(verdict.get("grounded", True)) or revised_accepted

    return {
        "answer": final,
        "citations": citations,
        "grounded": grounded,
        "unsupported": [] if grounded else (verdict.get("unsupported") or []),
        "revised": revised_accepted,
        "no_sources": False,
    }
