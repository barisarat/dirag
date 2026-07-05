# ragshelf

A small local CLI (the command is `ask`) that answers questions over folders of
PDFs (mine holds my technical book library), with page-cited answers. Personal
tool, shared as-is, no support. MIT.

```text
$ ask "how does the book motivate log compaction?"
+- answer ---------------------------------------------------------------+
| Log compaction keeps the latest value per key ... [1] ... segments are |
| rewritten in the background ... [2]                                    |
+------------------------------------------------------------------------+
sources
 1  designing-data-intensive-applications   page 456   "...compaction means..."
 2  designing-data-intensive-applications   page 458   "...merged segments..."
```

How it works: PDFs are parsed (PyMuPDF), chunked into page-anchored passages,
embedded locally (fastembed, bge-small-en-v1.5), and stored in one sqlite file
(sqlite-vec, exact brute-force KNN). A question is embedded, top passages are
retrieved and ranked, and the configured model drafts an answer that cites its
sources by page. A bounded verification pass checks the draft against the
cited passages and fails open (a flaky judge never blocks an answer).

## Setup

```bash
uv sync
# copy corpora.toml somewhere the tool finds it and edit the paths:
#   $ASK_CORPORA > ./corpora.toml > ~/.config/ask/corpora.toml
```

A corpus is a folder of PDFs plus the model that answers over it:

```toml
default = "library"

[library]
path = "~/vault/library"
retriever = "vector"
model = "openai:gpt-5-nano"     # or "local:qwen2.5:3b" for Ollama
```

The `model` reference is the only routing input - the provider is never chosen
from the environment:

- `openai:<model-id>` - an OpenAI-compatible cloud model. Requires
  `OPENAI_API_KEY` (a missing key is a hard error, not a silent fallback).
  Set `OPENAI_BASE_URL` for Groq/OpenRouter/other compatible endpoints.
- `local:<ollama-tag>` - a model served by your local Ollama. This path never
  reads the API key, so a corpus configured local stays on your machine even
  with keys exported. Ollama is only needed if you use a `local:` model.

Embeddings are always computed locally, whichever model answers.

### Setting the API key

An `openai:` corpus needs `OPENAI_API_KEY`. Instead of exporting it every
session, put it in a `.env` file once - the CLI reads it at startup. This is the
same on Linux, macOS, and Windows (it replaces the shell-specific `export` /
`$env:` step). A `local:` (Ollama) corpus needs no key and can skip this.

1. Create the file. The global location is read from any directory:

   ```bash
   mkdir -p ~/.config/ask
   nano ~/.config/ask/.env
   ```

2. Add the key (and optionally a compatible endpoint), then save:

   ```bash
   OPENAI_API_KEY=sk-...
   # OPENAI_BASE_URL=...   # optional: Groq/OpenRouter/other compatible endpoint
   ```

3. Restrict the file, since it holds a secret:

   ```bash
   chmod 600 ~/.config/ask/.env
   ```

4. Verify it is picked up (the model should no longer show `OPENAI_API_KEY not
   set`):

   ```bash
   uv run ask config
   ```

The file is looked for at `$ASK_ENV`, then `./.env`, then `~/.config/ask/.env`
(first value per key wins). A repo-local `./.env` works the same way. An explicit
`export OPENAI_API_KEY=...` still overrides the file, so clear a stale one (or
open a fresh shell) if `ask config` shows the wrong key.

### Corpus path

The `path` is any folder of PDFs, scanned recursively. To use an existing
Zotero library, point it at the storage folder (`~/Zotero/storage`); attachments
are indexed in place and each citation uses the PDF's file name. If you use
Zotero linked-file attachments, point `path` at that base directory instead.

Optional host tools: `ocrmypdf` + `tesseract` enable the OCR fallback for
scanned PDFs without a text layer.

## Use

```bash
ask "how are backups pruned?"     # answer over the default corpus
ask q "..." --corpus papers       # explicit corpus (or: -c papers)
ask index                         # build/refresh the index (incremental)
ask status                        # files indexed, chunks, last index, skips
ask config                        # resolved corpora (keys redacted)
```

Flags: `--k` result count, `--model provider:model` per-call override,
`--json` machine output, `--config` an explicit config path. In the shorthand
form the question comes first (`ask "..." --json`); a question that collides
with a command name needs the explicit form (`ask q "config"`).

## Adding books

Drop PDFs into the corpus path, then `ask index`. Indexing is incremental
(hash-diffed) and self-healing: only new or changed files are re-embedded,
files removed from disk have their chunks swept, and a re-run with no changes
is a near-instant no-op. A scanned PDF with no text layer is OCR'd if
`ocrmypdf` is present, otherwise reported as `no_text` (never silently
dropped).

Embedding is CPU-only and single-threaded by default, so a large library takes
a while - a thousand PDFs can run for several hours on one core. On a multi-core
box, parallelize with `ASK_EMBED_THREADS` (and a larger `ASK_EMBED_BATCH` to
feed it):

```bash
ASK_EMBED_THREADS=12 ASK_EMBED_BATCH=512 ask index
```

bge-small is a small model, so the speedup tapers off past a handful of threads;
leave a few cores for the rest of the machine. Indexing is incremental, so a
run interrupted partway is safely resumed by re-running.

## Building the index on another machine

Embedding a large book is memory-heavy (the model needs a few hundred MB). On
a small/low-RAM box the indexer can be OOM-killed mid-embed (exit 137). The
index is a single portable file, so build it where there is RAM and copy it
back:

1. On a high-RAM machine: clone the project, `uv sync`, put the same PDFs
   under the same corpus path, run `ask index`.
2. Copy `data/<corpus>.sqlite3` to the small machine's `data/` (the file is
   self-contained - the WAL is checkpointed on close).
3. On the small machine just query; a single-question embed is light. Keep the
   PDFs at the same path on both machines so a later `ask index` there is a
   no-op instead of sweeping the copied entries.

Knobs for a tight box: `ASK_EMBED_BATCH=16 ASK_EMBED_THREADS=1`.

## Notes

- Corpora and routing live in `corpora.toml`; a corpus is always resolved
  explicitly (named, or the configured default) - there is no auto-detection.
- The index lives at `data/<corpus>.sqlite3` (gitignored); override the
  directory with `ASK_DATA_DIR`.
- The embedding dimension is baked into the store, so changing the embedding
  model means re-indexing.
- `ASK_REASONING_EFFORT` - default `minimal` for gpt-5-family models; set
  empty for a model that rejects the parameter.
- `ASK_DEBUG=1` - show the (normally silent) fail-open judge diagnostics.
