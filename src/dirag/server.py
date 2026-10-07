"""The web app: a shelf of the library, search over it, and a reader for each book.

Standard library HTTP server. The page is static (static/index.html, app.js,
app.css); everything else is JSON or a file.

    GET  /                       the app
    GET  /static/<name>          a file from static/, by exact name only
    GET  /api/books              the shelf: every searchable book
    GET  /api/search?q=&mode=&rerank=&live=
                                 ranked passages; mode lexical|semantic|hybrid,
                                 rerank off|neural|llm, live=1 gives quick word matches
    GET  /api/answer?q=&mode=&rerank=
                                 the same ranked passages plus an answer: quotes
                                 verified against them, located on their pages
    GET  /api/features           {"llm": model name when the AI is available, else null}
    GET  /api/positions          {book id: page}
    GET  /api/bookmarks          [book id, ...]
    POST /api/position?id=       {"page": n}
    POST /api/bookmark?id=       {"on": bool}
    GET  /page?chunk=            PNG of a passage's page
    GET  /book?id=               the PDF, for the in-app reader
    GET  /api/library?scan=      the folder, whether it is fixed, and book counts;
                                 with scan, also PDFs new, changed and gone
    POST /api/library            {"path": folder} choose the library folder
    GET  /api/dirs?path=         subfolders of a folder under the browse root
    GET  /api/job                the indexing job record (see jobs.py)
    POST /api/job                {"action": "update" | "reindex"} start a job
    POST /api/job/stop           stop the running job

The library folder is read from config on every request, so choosing another
folder takes effect at once; each folder has its own index. With no folder or
no index yet, the shelf and search are empty.

Requests address books by index id and passages by chunk id, never by path, so
a request cannot name a file on disk. The folder picker accepts only folders
under the browse root. Static files are served from a fixed list made at
startup. PDFs are opened read-only.

The server binds 127.0.0.1 by default. It has no login, so bind it elsewhere
only on a network where every device is trusted.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pymupdf

from . import answer, config, index, jobs, llm, search, state

STATIC = Path(__file__).resolve().parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
FILES = {p.relative_to(STATIC).as_posix(): p for p in STATIC.rglob("*") if p.is_file() and p.suffix in TYPES}
# Page image resolution. The highlight is placed in page units, so it does not depend on this.
DPI = 130
SNIPPET_WIDTH = 260
RENDER_CACHE_MAX = 64
_render_cache = {}


def render_page(path, page):
    """PNG of one page, with a small cache: repeated searches land on the same pages."""
    key = (str(path), page)
    if key not in _render_cache:
        doc = pymupdf.open(path)
        try:
            body = doc.load_page(page - 1).get_pixmap(dpi=DPI).tobytes("png")
        finally:
            doc.close()
        if len(_render_cache) >= RENDER_CACHE_MAX:
            _render_cache.pop(next(iter(_render_cache)))
        _render_cache[key] = body
    return _render_cache[key]


def snippet(text, question, width=SNIPPET_WIDTH):
    """A window of the passage centred on the first query word found in it, else its head.

    The head is what a vector-only hit shows: it matched by meaning, so no word
    of the query need appear.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    lowered = flat.lower()
    words = "".join(c if c.isalnum() else " " for c in question.lower()).split()
    hits = [i for i in (lowered.find(w) for w in words if len(w) > 2) if i != -1]
    if not hits:
        return flat[:width].rstrip() + "..."
    start = max(0, min(hits) - width // 3)
    if start:
        space = flat.find(" ", start)
        start = space + 1 if space != -1 and space - start < 20 else start
    end = start + width
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        end = space if space != -1 else end
    return ("..." if start else "") + flat[start:end].strip() + ("..." if end < len(flat) else "")


class Handler(BaseHTTPRequestHandler):
    root = None

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type, cache=False):
        # The page cancels a search that a newer one replaces; its reply then has no reader.
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, code=200):
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _fail(self, code, message):
        self._send(code, message.encode(), "text/plain; charset=utf-8")

    def _db(self):
        """A read-only connection to the current folder's index, or None when there is none."""
        db = config.index_path(self.root) if self.root else None
        return search.connect(db) if db and db.exists() else None

    def _file(self, rel):
        """The PDF for a stored relative path, refused if the path climbs out of the library."""
        return None if Path(rel).is_absolute() or ".." in Path(rel).parts else self.root / rel

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def _id(self, params, name="id"):
        try:
            value = int(params.get(name, ["0"])[0])
        except ValueError:
            return 0
        return value if value > 0 else 0

    def _paths(self):
        """{book id: relative path} for every searchable book."""
        conn = self._db()
        if conn is None:
            return {}
        try:
            return {row["id"]: row["path"] for row in conn.execute("SELECT id, path FROM books WHERE status = 'ok'")}
        finally:
            conn.close()

    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)
        self.root = config.library()
        routes = {"/api/books": self.books, "/api/search": self.search, "/api/positions": self.positions,
                  "/api/bookmarks": self.bookmarks, "/page": self.image, "/book": self.pdf,
                  "/api/library": self.library, "/api/dirs": self.dirs, "/api/job": self.job,
                  "/api/features": self.features, "/api/answer": self.answer}
        if url.path == "/":
            return self.static("index.html")
        if url.path.startswith("/static/"):
            return self.static(url.path[len("/static/"):])
        if url.path in routes:
            return routes[url.path](params)
        self._fail(404, "not found")

    def do_POST(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)
        self.root = config.library()
        routes = {"/api/position": self.set_position, "/api/bookmark": self.set_bookmark,
                  "/api/library": self.set_library, "/api/job": self.start_job, "/api/job/stop": self.stop_job}
        if url.path in routes:
            return routes[url.path](params)
        self._fail(404, "not found")

    def static(self, name):
        path = FILES.get(name)
        if path is None:
            return self._fail(404, "not found")
        self._send(200, path.read_bytes(), TYPES[path.suffix], cache=name.startswith("vendor/"))

    def books(self, params):
        conn = self._db()
        if conn is None:
            return self._json([])
        try:
            rows = conn.execute("SELECT id, path, title, author, year, pages FROM books WHERE status = 'ok'").fetchall()
        finally:
            conn.close()
        cards = state.cards()
        shelf = []
        for row in rows:
            card = cards.get(row["path"], {})
            shelf.append({"id": row["id"], "title": card.get("title") or row["title"] or "(untitled)",
                          "subtitle": card.get("subtitle") or "",
                          "authors": card.get("authors") or ([row["author"]] if row["author"] else []),
                          "year": card.get("year") or row["year"], "pages": card.get("pages") or row["pages"]})
        shelf.sort(key=lambda b: b["title"].lower())
        self._json(shelf)

    def positions(self, params):
        paths, saved = self._paths(), state.positions()
        self._json({book_id: saved[path] for book_id, path in paths.items() if path in saved})

    def bookmarks(self, params):
        self._json(self._marked(self._paths(), state.bookmarks()))

    def _marked(self, paths, marked):
        ids = {path: book_id for book_id, path in paths.items()}
        return [ids[path] for path in marked if path in ids]

    def set_position(self, params):
        try:
            page = int(self._body()["page"])
        except (ValueError, KeyError, TypeError):
            return self._fail(400, "bad position")
        path = self._paths().get(self._id(params))
        if path is None or page < 1:
            return self._fail(400, "bad position")
        state.set_position(path, page)
        self._json({"ok": True})

    def set_bookmark(self, params):
        try:
            on = bool(self._body()["on"])
        except (ValueError, KeyError, TypeError):
            return self._fail(400, "bad bookmark")
        paths = self._paths()
        path = paths.get(self._id(params))
        if path is None:
            return self._fail(400, "bad bookmark")
        self._json(self._marked(paths, state.set_bookmark(path, on)))

    def search(self, params, with_answer=False):
        question = (params.get("q", [""])[0] or "").strip()
        live = params.get("live", ["0"])[0] == "1" and not with_answer
        mode = params.get("mode", ["hybrid"])[0]
        rerank = params.get("rerank", ["off"])[0]
        if mode not in search.MODES or rerank not in search.RERANKS:
            return self._fail(400, "bad mode")
        empty = {"results": [], "ms": 0, "dpi": DPI, **({"answer": {"summary": "", "quotes": []}} if with_answer else {})}
        if len(question) < 2:
            return self._json(empty)
        started = time.monotonic()
        conn = self._db()
        if conn is None:
            return self._json(empty)
        try:
            rows = search.passages(conn, question, mode, rerank, live)
            composed = answer.compose(question, rows, self._file) if with_answer else None
        except llm.LLMError as exc:
            return self._fail(502, str(exc))
        finally:
            conn.close()
        results = [{"chunk_id": row["id"], "book_id": row["book_id"], "book": row["book"], "year": row["year"],
                    "section": row["section"], "page": row["page"], "snippet": snippet(row["text"], question),
                    "rects": [json.loads(row["bbox_json"])] if row["bbox_json"] else [],
                    "dropped": bool(row.get("dropped"))} for row in rows]
        self._json({"results": results, "dpi": DPI, "ms": round((time.monotonic() - started) * 1000),
                    **({"answer": composed} if with_answer else {})})

    def answer(self, params):
        if not llm.enabled():
            return self._fail(400, f"AI is off: Ollama with {llm.MODEL} is not available")
        return self.search(params, with_answer=True)

    def image(self, params):
        conn = self._db()
        if conn is None:
            return self._fail(404, "no index")
        try:
            row = conn.execute("SELECT c.page, b.path FROM chunks c JOIN books b ON b.id = c.book_id WHERE c.id = ?",
                               (self._id(params, "chunk"),)).fetchone()
        finally:
            conn.close()
        path = self._file(row["path"]) if row else None
        if path is None:
            return self._fail(404, "unknown passage")
        try:
            body = render_page(path, row["page"])
        except Exception as exc:
            return self._fail(404, f"cannot open {row['path']}: {exc}")
        self._send(200, body, "image/png")

    def pdf(self, params):
        rel = self._paths().get(self._id(params))
        path = self._file(rel) if rel else None
        if path is None:
            return self._fail(404, "unknown book")
        try:
            size = path.stat().st_size
            handle = path.open("rb")
        except OSError as exc:
            return self._fail(404, f"cannot open {rel}: {exc}")
        with handle:
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
            self.end_headers()
            try:
                while block := handle.read(256 * 1024):
                    self.wfile.write(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def library(self, params):
        """The folder and its indexed counts. With ?scan=1, also the PDFs that are new, changed or gone."""
        info = {"path": str(self.root) if self.root else None, "fixed": config.FIXED, "browse": str(config.BROWSE_ROOT)}
        if self.root is None or not self.root.is_dir():
            return self._json(info)
        conn = self._db()
        known = {}
        if conn is not None:
            try:
                known = {row["path"]: row for row in conn.execute("SELECT path, status, size, mtime FROM books")}
            finally:
                conn.close()
        info.update(books=sum(1 for r in known.values() if r["status"] == "ok"),
                    no_text=sum(1 for r in known.values() if r["status"] == "no_text"))
        if params.get("scan"):
            on_disk = set(index.find_pdfs(self.root))
            changed = 0
            for rel in on_disk & known.keys():
                try:
                    stat = (self.root / rel).stat()
                except OSError:
                    continue
                changed += (known[rel]["size"], known[rel]["mtime"]) != (stat.st_size, stat.st_mtime)
            info.update(new=len(on_disk - known.keys()), changed=changed, gone=len(known.keys() - on_disk))
        self._json(info)

    def _browsable(self, value):
        """`value` as a resolved folder under the browse root, or None."""
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            return None
        return path if path.is_dir() and path.is_relative_to(config.BROWSE_ROOT) else None

    def dirs(self, params):
        path = self._browsable(params.get("path", [""])[0] or str(self.root or config.BROWSE_ROOT)) or config.BROWSE_ROOT
        try:
            children = sorted((c for c in path.iterdir() if c.is_dir() and not c.name.startswith(".")), key=lambda c: c.name.lower())
            pdfs = sum(1 for c in path.iterdir() if c.is_file() and c.suffix.lower() == ".pdf")
        except OSError as exc:
            return self._fail(403, str(exc))
        self._json({"path": str(path), "parent": str(path.parent) if path != config.BROWSE_ROOT else None,
                    "dirs": [c.name for c in children], "pdfs": pdfs})

    def set_library(self, params):
        if config.FIXED:
            return self._fail(403, "the library folder is fixed by DIRAG_LIBRARY")
        if jobs.running():
            return self._fail(409, "an indexing job is running")
        try:
            path = self._browsable(str(self._body()["path"]))
        except (ValueError, KeyError, TypeError):
            path = None
        if path is None:
            return self._fail(400, "not a folder under the browse root")
        config.set_library(path)
        self._json({"path": str(path)})

    def features(self, params):
        self._json({"llm": llm.MODEL if llm.enabled() else None})

    def job(self, params):
        self._json(jobs.status())

    def start_job(self, params):
        try:
            action = self._body()["action"]
        except (ValueError, KeyError, TypeError):
            action = None
        if action not in ("update", "reindex") or self.root is None:
            return self._fail(400, "bad job")
        try:
            jobs.start(action, self.root)
        except RuntimeError as exc:
            return self._fail(409, str(exc))
        self._json(jobs.status())

    def stop_job(self, params):
        jobs.stop()
        self._json(jobs.status())


def serve(host="127.0.0.1", port=8008, browser=True):
    """Run the app. Prints where it runs and what the GPU and AI status is; opens the browser."""
    from . import embed
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(f"Cannot listen on {host}:{port}: {exc.strerror}. Is dirag already running? Try --port.")
    address = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    gpu = embed.gpu_name()
    ai = llm.status()
    print(f"dirag on {address}", flush=True)
    print(f"  GPU   {gpu or 'none: indexing runs on the CPU'}", flush=True)
    print(f"  AI    {llm.MODEL} via Ollama" if ai == "ready" else
          f"  AI    pulling {llm.MODEL}" if ai == "missing" else
          f"  AI    off: Ollama not found at {llm.URL}", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  bound beyond loopback: there is no login", file=sys.stderr)
    llm.pull_in_background()
    if browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, (address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
