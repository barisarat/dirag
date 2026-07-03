"""Shared lexical helpers for the coverage/overlap scoring paths.

`tokenize` is used when scoring query/context overlap in answer ranking, so the
stop-word set and tokenizer live here to keep every caller in lock-step.
"""

import re

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "make",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "use",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(value):
    if not value:
        return []

    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-+.]*", value.lower())

    return [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
