"""Tolerant JSON extraction from model output.

Local models wrap JSON in prose or code fences, so callers can't json.loads
directly. `extract_json` pulls the first top-level JSON value (object or array) -
whichever bracket opens first wins, so a {...} containing an array isn't mistaken
for the inner list, and a bare [...] is parsed as an array. Returns `fallback` on
any failure, so the judge/extraction loops fail open rather than raise.
"""

import json


def extract_json(text, fallback=None):
    if not text:
        return fallback

    candidates = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append((start, text[start:end + 1]))

    # Whichever bracket type opens earliest in the text is the outer value.
    candidates.sort(key=lambda c: c[0])
    for _, fragment in candidates:
        try:
            return json.loads(fragment)
        except (ValueError, TypeError):
            continue

    return fallback
