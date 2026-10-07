# CLAUDE.md

dirag is a public, local-first tool: index a folder of PDF books, search it, and
read the page each result comes from. `README.md` is usage for people; this file
is how to work on the code.

## Writing rules

- **Public project.** No personal names, machines, paths, libraries or usage
  anywhere: code, comments, docs, examples, test data.
- **Current state, procedural.** Comments and docs say what the code does and
  how to use it now. No history, no "used to", no reasons that only made sense
  for an earlier version. Git keeps the story.
- **ASCII only** in code comments and Markdown. Check with
  `LC_ALL=C grep -rn '[^ -~]' --include='*.py' --include='*.md' --include='*.js' --include='*.css' --include='*.html' .`
- **Minimal UI text.** Labels are one or two words; no instructions on screen.
  Icons are Bootstrap Icons 1.11.3, paths copied verbatim from the upstream file
  into `static/icons.js` with the icon's name beside each; never draw one.

## Layout

```text
src/dirag/
  cli.py         dirag [serve] | index | find | where | toc
  config.py      DIRAG_HOME, DIRAG_STATE, the library folder, per-folder index path
  jobs.py        the indexing job: job.json, start/stop from the app, stop checks
  toc.py         a PDF's table of contents -> page-partitioned sections
  index.py       update | reindex | rechunk: PDFs -> sections -> passages -> sqlite
  embed.py       the local models' device (GPU when it works, else CPU), embeddings
  search.py      modes (lexical, semantic, hybrid), rerank hook, per-book cap, chapters
  rerank.py      neural (fastembed cross-encoder) and llm rerankers
  llm.py         Ollama client: live availability, model pull, chat with an 8k context
  answer.py      quotes chosen by the model, verified, located on their pages
  state.py       positions, bookmarks, cards; per-key writes
  server.py      stdlib HTTP server and the JSON API
  static/        index.html, app.js, app.css, icons.js, favicon.svg
    vendor/      pdf.js (pdf.min.mjs, pdf.worker.min.mjs, LICENSE.pdfjs)
```

## Invariants

- **Books are keyed by path relative to the library folder**, in the index and
  in every state file. Index ids are for requests only; they change on rebuild.
- **A request never names a file.** Books by id, passages by chunk id, static
  files from the list built at startup.
- **State writes are per key**, merged into the file on disk, written through a
  temp file and rename. Never accept a whole document from a client.
- **The stored passage text is verbatim.** The chapter header is embedded with
  the passage but never stored on it or shown.
- **Two BM25 indexes**: `chunks_fts` (passage text) and `sections_fts` (section
  titles). Do not fold titles into the passage index.
- **Pipeline order**: retrieve -> dedupe -> rerank -> per-book cap. The cap
  comes last so it applies to the final order.
- **A language model never writes text shown as a source.** The llm reranker
  returns passage numbers only; numbers out of range or repeated are ignored,
  and the passages it leaves out stay in the list marked `dropped`. An answer
  quote is shown only as the passage's own text for the span it matched (on
  letters and digits); the model's response (one to three sentences) loses
  citations to anything not quoted and is dropped when none remain.
- **Marks are rectangles in PDF points** (`rects`): a result's passage bbox, or
  an answer quote's line boxes found through the page's words. The page image
  and the reader draw the same list.
- **One install, no setup.** The dependencies are the GPU build of onnxruntime
  (CPU build on macOS); `embed.build` uses the GPU only when nvidia-smi names
  one and a first inference succeeds without onnxruntime falling back, and
  otherwise runs on the CPU with the CPU provider named explicitly. The AI
  features exist only while Ollama answers with the model; nothing else
  depends on them.
- **No build step.** The browser loads `static/` as is; pdf.js is vendored. To
  update pdf.js, copy `build/pdf.min.mjs`, `build/pdf.worker.min.mjs` and
  `LICENSE` from the `pdfjs-dist` npm package into `static/vendor/`.
- **Indexing is resumable**: one commit per book; a book is skipped when its
  size and mtime match, else when its content hash matches; cached page blocks
  in `pages` mean `--rechunk` never re-parses.
- **One indexing job at a time**, recorded in `job.json` by the indexer itself,
  so the app shows and stops a run started from either place. Stop is a request
  checked before each book, after each page read and after each embedding
  batch; the book in progress rolls back. Liveness is checked through /proc
  (not a zombie, still a dirag indexer), so a crashed run reads as failed.
- **One index per library folder** (`indexes/<name>-<hash>.sqlite3`); switching
  folders never touches another folder's index.
- **The folder picker is confined** to the browse root, and `DIRAG_LIBRARY`
  disables changing the folder from the app.

## Running from a checkout

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/dirag
```

Release: `uv build`, then `uv publish` (version in `pyproject.toml` and
`src/dirag/__init__.py`). Check the wheel carries `static/` and `static/vendor/`.
