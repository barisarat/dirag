"""The answer: quotes chosen by a language model, verified against the passages and located on the page.

The model sees the top passages of a search and returns passage numbers with
the words it quotes from each, plus a response of one to three sentences citing
them by number. Nothing
it writes is shown as a quote:

  - Each quote is matched against its passage on letters and digits only, so
    case, punctuation, spacing and line-break hyphens do not matter. A quote
    with "..." is matched part by part, in order. A quote that does not match
    is dropped.
  - The text shown for a quote is the passage's own text for the matched span.
  - The quote is located on its page through the page's words, giving one
    rectangle per line for the reader and the page image to mark.
  - Citations in the response to passages with no kept quote are removed, the
    rest renumbered to the quotes. The response is kept only when at least one
    citation remains; otherwise only the quotes are shown. With no kept quote the answer
    is empty.
"""

import json
import re

import pymupdf

from . import llm

DEPTH = 10
MAX_QUOTES = 5
MIN_ALNUM = 15

SYSTEM = ("You answer a question using only the numbered book passages given. Reply with JSON only:\n"
          '{"quotes": [{"n": <passage number>, "text": "<words copied exactly from that passage>"}], '
          '"summary": "<one to three sentences answering the question, each citing passages like [2]>"}\n'
          "Copy every quote word for word from its passage; use ... only to skip words inside a quote. "
          "Cite only passages you quoted. "
          f"Each quote is one to three sentences. Order quotes from most to least useful; at most {MAX_QUOTES}. "
          'If no passage answers the question, reply {"quotes": [], "summary": ""}.')


def _alnum(text):
    """Letters and digits of `text`, lowercased, with the index in `text` of each one."""
    chars, where = [], []
    for i, c in enumerate(text):
        if c.isalnum():
            chars.append(c.lower())
            where.append(i)
    return "".join(chars), where


def locate(quote, text):
    """(start, end) of `quote` within `text`, matched on letters and digits, or None."""
    haystack, where = _alnum(text)
    parts = [p for p in (_alnum(part)[0] for part in re.split(r"\.\.\.|\u2026", quote)) if p]
    if not parts or sum(len(p) for p in parts) < MIN_ALNUM:
        return None
    start, at = None, 0
    for part in parts:
        found = haystack.find(part, at)
        if found == -1:
            return None
        start = found if start is None else start
        at = found + len(part)
    return where[start], where[at - 1] + 1


def page_rects(path, page_number, span_text):
    """Line rectangles of `span_text` on a page, in PDF points, found through the page's words."""
    doc = pymupdf.open(path)
    try:
        words = doc.load_page(page_number - 1).get_text("words")
    finally:
        doc.close()
    joined, owner = [], []
    for index, word in enumerate(words):
        letters = _alnum(word[4])[0]
        joined.append(letters)
        owner.extend([index] * len(letters))
    needle = _alnum(span_text)[0]
    found = "".join(joined).find(needle)
    if found == -1 or not needle:
        return []
    lines = {}
    for index in sorted(set(owner[found:found + len(needle)])):
        x0, y0, x1, y1, _, block, line, _ = words[index]
        box = lines.setdefault((block, line), [x0, y0, x1, y1])
        box[:] = [min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
    return [[round(v, 2) for v in box] for box in lines.values()]


def compose(question, rows, file_of):
    """{"summary", "quotes"} for `rows` (search rows, best first). `file_of(rel)` gives a book's PDF path.

    Raises llm.LLMError when the model call fails.
    """
    top = rows[:DEPTH]
    listing = "\n\n".join(f"[{n}] {row['book']} > {row['section'] or ''}, p. {row['page']}\n{row['text']}"
                          for n, row in enumerate(top, 1))
    reply = llm.chat_json(SYSTEM, f"Question: {question}\n\nPassages:\n\n{listing}")
    quotes, numbers = [], {}
    for item in reply.get("quotes", []) if isinstance(reply, dict) else []:
        try:
            n, said = int(item["n"]), str(item["text"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= n <= len(top) or len(quotes) >= MAX_QUOTES:
            continue
        row = top[n - 1]
        span = locate(said, row["text"])
        if span is None:
            continue
        text = row["text"][span[0]:span[1]]
        if any(q["chunk_id"] == row["id"] and q["text"] == text for q in quotes):
            continue
        path = file_of(row["path"])
        quotes.append({"n": len(quotes) + 1, "chunk_id": row["id"], "book_id": row["book_id"], "book": row["book"],
                       "year": row["year"], "section": row["section"], "page": row["page"], "text": text,
                       "rects": (page_rects(path, row["page"], text) if path else []) or _bbox(row)})
        numbers.setdefault(n, quotes[-1]["n"])
    summary = str(reply.get("summary") or "").strip() if quotes else ""
    summary = re.sub(r"\s*\[(\d+)\]", lambda m: f" [{numbers[int(m.group(1))]}]" if int(m.group(1)) in numbers else "",
                     summary).strip()
    if not re.search(r"\[\d+\]", summary):
        summary = ""
    return {"summary": summary, "quotes": quotes}


def _bbox(row):
    return [json.loads(row["bbox_json"])] if row["bbox_json"] else []
