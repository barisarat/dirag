# CLAUDE.md - working in the `ask` repo

`ask` is a small local CLI for grounded, page-cited question-answering over
folders of PDFs. Personal tool, published as-is (MIT): no PyPI release, no
support commitments, README stays short and factual. This file is guidance for
anyone (human or AI assistant) changing the code.

## What this is

One CLI over N configured corpora. A corpus = a folder of PDFs + the model that
answers over it, declared in `corpora.toml`:

- `ask "<question>"` answers over the default corpus; `ask q "..." -c <name>`
  targets a named one. `ask index` / `ask status` / `ask config` manage it.
- Retrieval: fastembed (bge-small-en-v1.5, 384-dim, always local) + sqlite-vec
  (one sqlite file per corpus, brute-force exact KNN, incremental upsert).
- Answering: the corpus's configured model drafts from retrieved passages with
  [n] page citations; a bounded fail-open judge verifies the draft.

## Model routing (core invariant)

Routing is a pure function of configuration. Each corpus declares
`model = "provider:model"`:

- `local:<ollama-tag>`: talks to Ollama; NEVER reads OPENAI_API_KEY. The cloud
  path is unreachable for a local corpus (see `tests/test_routing.py`).
- `openai:<model-id>`: requires OPENAI_API_KEY; a missing key raises a setup
  error. No silent fallback in either direction.

The environment supplies credentials and endpoints (OPENAI_API_KEY,
OPENAI_BASE_URL, OLLAMA_BASE_URL) - never the choice of provider. The only
runtime routing input besides config is the explicit `--model provider:model`
override.

## House rules

- The corpus is always resolved explicitly: a named corpus, or the configured
  `default` (a single defined corpus is the implicit default). No
  auto-detection, no fuzzy inference.
- Retrieval and rendering are deterministic; the LLM answers from retrieved
  passages and never invents facts or writes markup.
- Judge loops are bounded and fail OPEN: a flaky judge never blocks an answer.
- Indexing is idempotent (hash-diffed) and self-healing (orphan sweep); any
  re-run converges. Never silently drop a file - report skips.
- Plain ASCII everywhere (copy, comments, docs): no em/en dashes, no curly
  quotes, no ellipsis characters. Use - and ... instead.
- Docs and comments describe the system as it is; they do not argue design
  alternatives.

## Stack

- Python >= 3.12, uv + pyproject.toml, src/ layout, console script `ask`.
- typer (CLI) + rich (output). tomllib (stdlib) for config.
- pymupdf (parse) + fastembed (embeddings) + sqlite-vec (store). ocrmypdf +
  tesseract are optional OCR host tools.
- LLM: an OpenAI-compatible client (Ollama local + OpenAI-compatible cloud),
  fed provider/model from `corpora.toml` + env.
- No server, no web framework, no SQLAlchemy. Single process, single user,
  local files.

## Conventions

- Add a dependency by editing `pyproject.toml`, then `uv sync`; do not hand-edit
  `uv.lock`.
- Run the suite with `uv run pytest -q`. Index/store/chunker/routing tests run
  with no network and no LLM (a fake embedder; the embedding model downloads
  only on a real `ask index`).

## Out of scope (do not build)

Hybrid lexical+vector retrieval; ANN/Qdrant/FAISS/LanceDB; compare/write skills;
any TUI/web/daemon or auto-reindex; non-PDF formats; per-document privacy tagging
(privacy is per corpus, by folder); PyPI packaging.
