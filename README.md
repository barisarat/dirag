# diRAG

Search a folder of PDF books, read the page each result comes from, and get
answers quoted from the books.

dirag indexes a library of PDFs once, then answers a query with ranked passages:
book, chapter, page and a snippet. Each result opens the rendered page with the
passage marked on it, or the whole book in the built-in reader at that page.

- Passages carry their chapter, taken from each PDF's table of contents.
- Three retrieval modes: lexical (BM25), semantic (embeddings), and hybrid
  (both, fused by rank). An optional rerank reads query and passage together:
  neural (a local cross-encoder) or an LLM.
- Results are capped per book, so one long series does not fill the list.
- The AI answer quotes the books: a language model picks the passages and the
  words, dirag checks every quote against the text and shows it in the book's
  own words, marked on its page.
- The reader remembers the page you stopped on in every book, and its text can
  be selected and copied.
- Everything runs locally. No account, no network after the first downloads.

## Start

```sh
uvx dirag
```

That needs only [uv](https://docs.astral.sh/uv/). The browser opens on the app:
choose the library folder (the folder button at the top of the shelf), press
**Scan**, then **Index new**. Subfolders are included.

dirag uses two things when they are there, with nothing to set:

| When present | Gives |
|---|---|
| An NVIDIA GPU | indexing in minutes per hundred books instead of hours; a faster neural rerank |
| [Ollama](https://ollama.com), running | the LLM rerank and the AI answer; dirag pulls `qwen2.5:7b-instruct` through it on the first start |

Without them, search, the neural rerank and the reader work as usual; the AI
button and the LLM rerank are hidden. The terminal shows what was found:

```text
dirag on http://127.0.0.1:8008
  GPU   NVIDIA GeForce RTX 3060
  AI    qwen2.5:7b-instruct via Ollama
```

The first start downloads the packages (about 2.5 GB on Linux and Windows,
GPU libraries included), the embedding and rerank models (about 250 MB) and,
with Ollama, the language model (about 4.7 GB). To keep dirag installed rather
than run it through uvx: `uv tool install dirag`, then `dirag`.

## Indexing

Indexing is long by design: every page is parsed and every passage embedded.
**Scan** looks through the folder for PDFs that are new, changed or gone;
**Index new** appears when it finds any, adds the new and changed PDFs and
drops the deleted ones. **Reindex all** parses and embeds everything again.
While a run is going the library view shows its progress and a time estimate,
and for the current book the pages read or passages embedded and the minutes
spent on it; **Stop** appears only then. Stopping keeps every finished book,
and the next run continues from there. Search works on the finished books while
a run is going.

## Search

| Key | Does |
|---|---|
| typing | lists quick exact-word matches under the box; pick one (click, or arrows and Enter) to open its page |
| Enter, or the search button | searches with the chosen mode and rerank |
| Ctrl+Enter, or the AI button | answers from the books (see below) |

The mode and rerank sit beside the search box:

| Mode | Finds |
|---|---|
| Hybrid | both of the below, fused by rank (the default) |
| Lexical | the exact words |
| Semantic | passages close in meaning, without the words |

| Rerank | Does |
|---|---|
| No rerank | keeps the retrieval order |
| Neural | scores the top 30 with a local cross-encoder; about a second on a CPU |
| LLM | asks the language model to order the top 20 and leave out the ones that do not help; those stay in the list, faded |

The line under the search box names what produced the results, with the count
and the time. While a search runs, the current results fade under a moving bar.

## Answer

Ctrl+Enter or the AI button runs the search, then asks the model which passages
answer the question and which words to quote from each. Only what the model
chose is shown, as one card marked with the AI icon:

- its response, one to three sentences citing the quotes by number; citations
  to anything not quoted are removed, and a response left with none is dropped;
- the quotes, as quotations in the books' own words, each with its source: the
  source opens the reader scrolled to the quote, marked on the page, and the
  page icon shows the page image in place.

Every quote is matched against its passage before it is shown; a quote that is
not in the text is dropped. When no passage answers the question, the answer
says so. The response is the model's reading, not the books; the quotes and
their pages are what to check.

## Terminal

```sh
dirag --port 9000 --no-browser   # serve on another port, without opening the browser
dirag index --library ~/Books    # choose the folder (remembered) and index it
dirag index                      # index new and changed PDFs, drop deleted ones
dirag index --reindex            # parse and embed every PDF again
dirag index --rechunk            # rebuild passages and vectors from cached pages
dirag find "martingale" --mode semantic --rerank neural   # passages
dirag where "instrumental variables"                      # chapters
dirag toc book.pdf               # the chapter map dirag reads from one PDF
```

Ctrl-C stops an index run the same way the stop button does; press it twice to
stop at once.

## Settings

All optional.

| Variable | Default | Sets |
|---|---|---|
| `DIRAG_HOME` | `~/.local/share/dirag` | indexes (one per library folder), models, the job record |
| `DIRAG_STATE` | `DIRAG_HOME` | `config.json`, `positions.json`, `bookmarks.json`, `cards.json` |
| `DIRAG_LIBRARY` | chosen in the app | fixes the library folder; the app then cannot change it |
| `DIRAG_BROWSE_ROOT` | the home directory | where the folder picker may browse |
| `DIRAG_LLM_MODEL` | `qwen2.5:7b-instruct` | the Ollama model |
| `DIRAG_LLM_URL` | `http://127.0.0.1:11434` | the Ollama server, which can be another machine |
| `DIRAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-12-v2` | the cross-encoder; `BAAI/bge-reranker-base` is larger and slower |
| `DIRAG_DEVICE` | detected | `cpu` keeps the local models off the GPU |

Books are stored by their path inside the library, so the folder can move as
long as its contents keep their relative paths; a moved folder gets a new index
unless its index file is renamed to match.

`cards.json` overrides how a book is shown on the shelf. Keys are paths relative
to the library:

```json
{"statistics/all-of-statistics.pdf": {"title": "All of Statistics", "subtitle": "A Concise Course",
  "authors": ["Larry Wasserman"], "year": 2004, "pages": 442}}
```

## Running as a service

Install it with `uv tool install dirag`, then a systemd unit:

```ini
[Unit]
Description=dirag
After=network-online.target

[Service]
ExecStart=%h/.local/bin/dirag --no-browser
Restart=on-failure

[Install]
WantedBy=default.target
```

As a user unit (`~/.config/systemd/user/dirag.service`, then
`systemctl --user enable --now dirag`). Stopping the service stops a running
index job the same way the stop button does.

## Security

The server binds 127.0.0.1 and has no login. Anyone who can reach the port can
read every book in the library, choose another folder under the browse root and
start indexing. Bind another address (`--host`) only on a network where every
device is trusted, and set `DIRAG_LIBRARY` to fix the folder.

## License

MIT. pdf.js (Apache-2.0) and Bootstrap Icons (MIT) are included; see
`src/dirag/static/vendor/LICENSE.pdfjs`.
