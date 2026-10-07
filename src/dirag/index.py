"""Build the index: PDFs -> chapter map -> page-anchored passages -> one sqlite file.

The file holds book metadata, the chapter map, the passages, a BM25 index over
passage text, a BM25 index over section titles, the passage vectors, and the
extracted page blocks.

    Books are keyed by their path RELATIVE to the library folder, so the library
    can move as long as its inside stays the same.

    Passages carry their chapter. What is embedded is "<book> > <chapter path>"
    followed by the passage; what is stored and shown is the passage alone.

    The two BM25 indexes are separate. chunks_fts holds passage text only, so a
    chapter title does not make every passage under it match. sections_fts holds
    the titles.

    Passages never cross a page, so each has one page number and one bounding
    box, and can be marked where it sits on the page.

    Page blocks are cached in `pages`, so a rechunk re-embeds without re-parsing.

Three actions:

    update     index new and changed PDFs, and drop books whose file is gone.
               A book whose size and modification time are unchanged is skipped
               without reading it; otherwise its content hash decides.
    reindex    parse and embed every PDF again.
    rechunk    rebuild passages and vectors from the cached page blocks.

Each book commits on its own, so an interrupted run keeps every finished book.
Progress goes to the job record (see jobs.py). Only one run at a time.
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import sqlite_vec

from . import embed, jobs
from .toc import chapters_of, sections_of

# Passage size: large enough to hold an argument, small enough that the box on
# the page stays tight.
TARGET_CHARS = 900
OVERLAP_CHARS = 150
# Below this many extracted characters a book has no text layer. It is recorded
# with status 'no_text' and left out of search.
NO_TEXT_THRESHOLD = 2000
NO_CHAPTERS = "(no chapter map)"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT, author TEXT, year INTEGER,
    pages INTEGER, hash TEXT, size INTEGER, mtime REAL,
    sections INTEGER, chunks INTEGER,
    status TEXT, error TEXT, indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    ord INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    title TEXT NOT NULL,
    path_text TEXT NOT NULL,
    chapter TEXT NOT NULL,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sections_book ON sections(book_id, start_page);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    section_id INTEGER,
    page INTEGER NOT NULL,
    bbox_json TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_book ON chunks(book_id, page);
CREATE TABLE IF NOT EXISTS pages (
    book_id INTEGER NOT NULL,
    page INTEGER NOT NULL,
    blocks_json TEXT NOT NULL,
    PRIMARY KEY (book_id, page)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts
    USING fts5(path_text, content='sections', content_rowid='id');
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def connect(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(books)")}
    for column, kind in (("size", "INTEGER"), ("mtime", "REAL")):
        if column not in columns:
            conn.execute(f"ALTER TABLE books ADD COLUMN {column} {kind}")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                 f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{embed.DIM}])")
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_model', ?)", (embed.MODEL,))
    conn.commit()
    return conn


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path, doc):
    """Title from the PDF's own metadata, else the filename. Author and year stay empty.

    An embedded title that is itself a filename, or has no letters, is ignored.
    cards.json in the state directory overrides any of these for display.
    """
    title = " ".join(((doc.metadata or {}).get("title") or "").split())
    if title.lower().endswith((".pdf", ".doc", ".docx", ".tex", ".dvi", ".indd")) or not any(c.isalpha() for c in title):
        title = ""
    return (title or Path(path).stem.replace("-", " ").replace("_", " ")).strip()[:300], None, None


def page_blocks(page):
    """Text blocks of one page in reading order (left column, then right), with bboxes."""
    raw = page.get_text("blocks")
    blocks = [b for b in raw if len(b) >= 5 and (b[4] or "").strip() and (len(b) <= 6 or b[6] == 0)]
    mid = page.rect.width / 2.0
    blocks.sort(key=lambda b: (0 if (b[0] + b[2]) / 2.0 < mid else 1, round(b[1], 1), b[0]))
    return [{"text": " ".join(b[4].split()), "bbox": [b[0], b[1], b[2], b[3]]} for b in blocks]


def union_bbox(bboxes):
    if not bboxes:
        return None
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


def window(blocks):
    """One page's blocks as ~TARGET_CHARS passages, each overlapping the previous by OVERLAP_CHARS."""
    passages, buf, boxes = [], "", []
    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue
        if buf and len(buf) + 1 + len(text) > TARGET_CHARS:
            passages.append({"text": buf, "bbox": union_bbox(boxes)})
            buf = f"{buf[-OVERLAP_CHARS:]} {text}".strip()
            boxes = [block["bbox"]]
        else:
            buf = f"{buf} {text}".strip() if buf else text
            boxes.append(block["bbox"])
    if buf:
        passages.append({"text": buf, "bbox": union_bbox(boxes)})
    return passages


def section_rows(doc):
    """The chapter map as rows for the sections table, one per leaf section."""
    leaves = sections_of(doc)
    if not leaves:
        return [{"ord": 0, "depth": 1, "title": NO_CHAPTERS, "path_text": NO_CHAPTERS,
                 "chapter": NO_CHAPTERS, "start_page": 1, "end_page": max(doc.page_count, 1)}]
    chapters = chapters_of(leaves)
    rows, index = [], 0
    for ordinal, leaf in enumerate(leaves):
        while index + 1 < len(chapters) and chapters[index + 1]["start_page"] <= leaf["start_page"]:
            index += 1
        rows.append({"ord": ordinal, "depth": leaf["depth"], "title": leaf["title"],
                     "path_text": " > ".join(leaf["path"]), "chapter": chapters[index]["title"],
                     "start_page": leaf["start_page"], "end_page": leaf["end_page"]})
    return rows


def store_pages(conn, book_id, blocks_by_page):
    conn.executemany("INSERT OR REPLACE INTO pages (book_id, page, blocks_json) VALUES (?,?,?)",
                     [(book_id, page, json.dumps(blocks)) for page, blocks in blocks_by_page.items()])


def load_pages(conn, book_id):
    """Cached blocks for a book as {page: [{text, bbox}]}, or {} if not cached."""
    rows = conn.execute("SELECT page, blocks_json FROM pages WHERE book_id = ? ORDER BY page", (book_id,)).fetchall()
    return {row["page"]: json.loads(row["blocks_json"]) for row in rows}


def purge_derived(conn, book_id):
    """Drop a book's sections, passages, vectors and FTS rows. The cached pages stay."""
    for row in conn.execute("SELECT id FROM chunks WHERE book_id = ?", (book_id,)):
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row["id"],))
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (row["id"],))
    for row in conn.execute("SELECT id FROM sections WHERE book_id = ?", (book_id,)):
        conn.execute("DELETE FROM sections_fts WHERE rowid = ?", (row["id"],))
    conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
    conn.execute("DELETE FROM sections WHERE book_id = ?", (book_id,))


def upsert_book(conn, rel, title, author, year, pages, digest, stat, status="ok", error=None):
    """Insert or update the book row and return its id."""
    values = (title, author, year, pages, digest, stat.st_size, stat.st_mtime, status, error, now())
    row = conn.execute("SELECT id FROM books WHERE path = ?", (rel,)).fetchone()
    if row:
        conn.execute("UPDATE books SET title=?, author=?, year=?, pages=?, hash=?, size=?, mtime=?, status=?, error=?,"
                     " indexed_at=? WHERE id=?", (*values, row["id"]))
        return row["id"]
    return conn.execute("INSERT INTO books (title, author, year, pages, hash, size, mtime, status, error, indexed_at,"
                        " sections, chunks, path) VALUES (?,?,?,?,?,?,?,?,?,?,0,0,?)", (*values, rel)).lastrowid


def remove_book(conn, book_id):
    purge_derived(conn, book_id)
    conn.execute("DELETE FROM pages WHERE book_id = ?", (book_id,))
    conn.execute("DELETE FROM books WHERE id = ?", (book_id,))


def needs_work(conn, root, rel, action):
    """(True, digest) when the book must be processed under `action`, else (False, None)."""
    row = conn.execute("SELECT hash, size, mtime, status FROM books WHERE path = ?", (rel,)).fetchone()
    if action == "rechunk":
        return (row is not None and row["status"] == "ok"), None
    if action == "reindex" or row is None:
        return True, None
    stat = (root / rel).stat()
    if row["size"] == stat.st_size and row["mtime"] == stat.st_mtime:
        return False, None
    digest = file_hash(root / rel)
    if digest == row["hash"]:
        conn.execute("UPDATE books SET size=?, mtime=? WHERE path=?", (stat.st_size, stat.st_mtime, rel))
        conn.commit()
        return False, None
    return True, digest


def index_book(conn, root, rel, action="update", digest=None, step=None):
    """Parse (or, for rechunk, reuse the cached pages), section, chunk, embed and store one book.

    `step(name, n, of)` hears each page read ('read') and each batch embedded ('embed').
    Returns 'ok', 'rechunk' or 'no_text'.
    """
    step = step or (lambda name, n, of: None)
    path = root / rel
    stat = path.stat()
    digest = digest or file_hash(path)
    existing = conn.execute("SELECT id FROM books WHERE path = ?", (rel,)).fetchone()

    doc = pymupdf.open(path)
    try:
        title, author, year = metadata(path, doc)
        rows = section_rows(doc)
        pages = doc.page_count
        cached = load_pages(conn, existing["id"]) if (existing and action == "rechunk") else {}
        blocks_by_page = cached or {}
        if not cached:
            for i in range(pages):
                blocks_by_page[i + 1] = page_blocks(doc.load_page(i))
                jobs.check()
                step("read", i + 1, pages)
    finally:
        doc.close()

    by_page = {page: section["ord"] for section in rows for page in range(section["start_page"], section["end_page"] + 1)}
    passages, total_chars = [], 0
    for page in sorted(blocks_by_page):
        for passage in window(blocks_by_page[page]):
            total_chars += len(passage["text"])
            passages.append({"page": page, "bbox": passage["bbox"], "text": passage["text"], "ord": by_page.get(page, 0)})

    if total_chars < NO_TEXT_THRESHOLD:
        book_id = upsert_book(conn, rel, title, author, year, pages, digest, stat, "no_text", "no text layer")
        purge_derived(conn, book_id)
        conn.execute("DELETE FROM pages WHERE book_id = ?", (book_id,))
        conn.execute("UPDATE books SET sections=0, chunks=0 WHERE id=?", (book_id,))
        conn.commit()
        return "no_text"

    book_id = upsert_book(conn, rel, title, author, year, pages, digest, stat)
    purge_derived(conn, book_id)
    if not cached:
        conn.execute("DELETE FROM pages WHERE book_id = ?", (book_id,))
        store_pages(conn, book_id, blocks_by_page)

    section_ids = {}
    for section in rows:
        cursor = conn.execute("INSERT INTO sections (book_id, ord, depth, title, path_text, chapter, start_page,"
                              " end_page) VALUES (?,?,?,?,?,?,?,?)",
                              (book_id, section["ord"], section["depth"], section["title"], section["path_text"],
                               section["chapter"], section["start_page"], section["end_page"]))
        section_ids[section["ord"]] = cursor.lastrowid
        conn.execute("INSERT INTO sections_fts (rowid, path_text) VALUES (?, ?)", (cursor.lastrowid, section["path_text"]))

    headers = {s["ord"]: f"{title} > {s['path_text']}" for s in rows}
    chunk_ids, to_embed = [], []
    for passage in passages:
        cursor = conn.execute("INSERT INTO chunks (book_id, section_id, page, bbox_json, text) VALUES (?,?,?,?,?)",
                              (book_id, section_ids.get(passage["ord"]), passage["page"],
                               json.dumps(passage["bbox"]) if passage["bbox"] else None, passage["text"]))
        chunk_ids.append(cursor.lastrowid)
        conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (cursor.lastrowid, passage["text"]))
        to_embed.append(f"{headers.get(passage['ord'], title)}\n\n{passage['text']}")

    batch = embed.batch_size()
    for start in range(0, len(to_embed), batch):
        vectors = embed.passages(to_embed[start:start + batch])
        conn.executemany("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                         [(chunk_ids[start + offset], json.dumps(vector)) for offset, vector in enumerate(vectors)])
        jobs.check()
        step("embed", min(start + batch, len(to_embed)), len(to_embed))

    conn.execute("UPDATE books SET sections=?, chunks=? WHERE id=?", (len(rows), len(passages), book_id))
    conn.commit()
    return "rechunk" if cached else "ok"


def find_pdfs(root):
    """Every PDF under the library folder, as sorted POSIX paths relative to it."""
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def run(root, db_path, action="update", limit=0):
    """Run `action` over the library at `root` into `db_path`. Returns a process exit code."""
    job = jobs.status()
    if job.get("state") == "running" and job.get("pid") != os.getpid():
        print(f"An indexing job is already running (pid {job['pid']}).", file=sys.stderr)
        return 1
    progress = jobs.Progress(action, root)
    jobs.catch_stop()
    try:
        return _run(root, db_path, action, limit, progress)
    except KeyboardInterrupt:
        progress.finish("stopped")
        print("\nstopped; finished books are kept", file=sys.stderr)
        return 130
    except BaseException as exc:
        progress.finish("failed", str(exc)[:300] or exc.__class__.__name__)
        raise


def _run(root, db_path, action, limit, progress):
    rels = find_pdfs(root)
    conn = connect(db_path)
    counts = {"ok": 0, "rechunk": 0, "skip": 0, "no_text": 0, "error": 0, "removed": 0}
    try:
        if not limit:
            on_disk = set(rels)
            for row in conn.execute("SELECT id, path FROM books").fetchall():
                if row["path"] not in on_disk:
                    remove_book(conn, row["id"])
                    counts["removed"] += 1
            conn.commit()
        rels = rels[:limit or None]

        todo = []
        for rel in rels:
            jobs.check()
            needed, digest = needs_work(conn, root, rel, action)
            if needed:
                todo.append((rel, digest, (root / rel).stat().st_size))
            else:
                counts["skip"] += 1
        progress.write(total=len(todo), bytes_total=sum(size for _, _, size in todo), counts=counts)
        if todo:
            embed.preflight()

        started = time.monotonic()
        for number, (rel, digest, size) in enumerate(todo, 1):
            jobs.check()
            print(f"  {number}/{len(todo)} {rel[:66]}", file=sys.stderr)
            progress.book(rel, size)
            begun = time.monotonic()

            def step(name, n, of):
                if progress.step(name, n, of):
                    print(f"      {'read' if name == 'read' else 'embedded'} {n}/{of} {'pages' if name == 'read' else 'passages'}"
                          f"  {(time.monotonic() - begun) / 60:.1f} min", end="\r", file=sys.stderr)

            try:
                counts[index_book(conn, root, rel, action, digest, step)] += 1
                print(file=sys.stderr)
            except KeyboardInterrupt:
                conn.rollback()
                raise
            except Exception as exc:
                counts["error"] += 1
                conn.rollback()
                print(f"      ERROR {str(exc)[:100]}", file=sys.stderr)
            progress.write(done=number, bytes_done=progress.job["bytes_done"] + size, counts=counts)
    finally:
        conn.rollback()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

    progress.finish("done")
    print("")
    print(f"processed    {counts['ok']} parsed, {counts['rechunk']} re-chunked, {counts['no_text']} without text,"
          f" {counts['error']} errors; {counts['skip']} unchanged, {counts['removed']} removed")
    print(f"elapsed      {(time.monotonic() - started) / 60:.1f} min")
    return 0
