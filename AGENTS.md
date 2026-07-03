# AGENTS.md

Short contract for anyone (human or AI assistant) changing `ask`. See CLAUDE.md
for the fuller version.

## Conventions

- Add a dependency by editing `pyproject.toml`, then `uv sync`; never edit
  `uv.lock` by hand. Run the suite with `uv run pytest -q`.
- Plain ASCII in all copy, comments, and docs (no em/en dashes, curly quotes,
  or ellipsis characters). Docs describe what is there, not design debates.

## Invariants

- A corpus = a folder of PDFs + `model = "provider:model"` in corpora.toml.
  Corpus resolution is explicit (named, or the configured default). No
  auto-detection.
- Routing is a pure function of config: `local:...` never reads OPENAI_API_KEY;
  `openai:...` requires it and fails loudly without it. No silent fallbacks
  (tests/test_routing.py).
- Embeddings are always local (fastembed); the store is sqlite-vec, one file
  per corpus at data/<corpus>.sqlite3.
- Judge loops are bounded and fail OPEN. Indexing is hash-diffed, idempotent,
  self-healing (orphan sweep), and never silently drops a file.
