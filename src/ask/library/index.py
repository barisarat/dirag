"""Incremental corpus indexing: diff, heal, sweep.

Walk the corpus for PDFs -> content hash; compare to the files table; parse/OCR/
chunk/embed/upsert new or changed files; sweep files gone from disk. Idempotent:
re-running with no changes is a near-instant no-op. Self-healing: a previously
errored file is retried on the next pass. Never silently drops a file - a scanned
PDF with no recoverable text is recorded as 'no_text' and reported.

The embed function and OCR function are injectable so the store/index tests run
deterministically with NO LLM and NO network (a fake embedder, a fake OCR).
"""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from . import chunker, ocr as ocr_mod, parser
from .embedder import embed_texts

# Total extracted characters below which a parse counts as "no usable text layer".
EMPTY_TEXT_THRESHOLD = 32
# Embed in bounded batches so peak memory stays low on small boxes (a 500-page
# book can be thousands of chunks) and so progress is observable mid-file.
EMBED_BATCH = int(os.getenv("ASK_EMBED_BATCH", "256"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_pdfs(root):
    """Sorted list of PDF paths under root (recursive, case-insensitive extension)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )


def scan_pdfs(root):
    """{absolute_path: content_hash} for every PDF under root (recursive)."""
    return {str(p.resolve()): sha256_file(p) for p in find_pdfs(root)}


def _total_chars(blocks):
    return sum(len((b.get("text") or "")) for b in blocks)


def build_chunks(blocks):
    """Group parsed blocks by page and window each page into passages."""
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)
    out = []
    for page in sorted(by_page):
        for passage in chunker.page_passages(by_page[page]):
            out.append({"page": page, "bbox": passage["bbox"], "text": passage["text"]})
    return out


def _embed_in_batches(texts, embed_fn, progress, label):
    """Embed texts in EMBED_BATCH-sized batches, reporting progress as it goes."""
    vectors = []
    total = len(texts)
    for batch in chunker.batched(texts, EMBED_BATCH):
        vectors.extend(embed_fn(batch))
        if progress:
            progress(f"    {label}: embedded {len(vectors)}/{total} chunks")
    return vectors


def _index_one(store, path, file_hash, embed_fn, ocr_fn, progress=None):
    """Parse (with OCR fallback), chunk, embed, and upsert one file. Returns its status."""
    name = Path(path).name
    mtime = Path(path).stat().st_mtime
    if progress:
        progress(f"    {name}: parsing")
    blocks = parser.parse_pdf(path)
    status = "ok"

    if _total_chars(blocks) < EMPTY_TEXT_THRESHOLD:
        ocred = ocr_fn(path) if ocr_fn else None
        if ocred:
            try:
                blocks = parser.parse_pdf(ocred)
                status = "ocr"
            finally:
                ocr_mod._safe_unlink(ocred)
        if _total_chars(blocks) < EMPTY_TEXT_THRESHOLD:
            # No recoverable text: record and report, keep no stale chunks.
            store.replace_file(
                path, file_hash, mtime, parser.page_count(path), "no_text", _now(), [], []
            )
            return "no_text"

    chunks = build_chunks(blocks)
    texts = [c["text"] for c in chunks]
    if progress:
        progress(f"    {name}: {len(texts)} chunks, embedding (batch {EMBED_BATCH})")
    vectors = _embed_in_batches(texts, embed_fn, progress, name) if texts else []
    if progress:
        progress(f"    {name}: writing {len(chunks)} chunks to the index")
    store.replace_file(
        path, file_hash, mtime, parser.page_count(path), status, _now(), chunks, vectors
    )
    return status


def build_index(store, root, *, embed_fn=embed_texts, ocr_fn=None, use_ocr=True, progress=None):
    """Run one diff-and-heal pass over root. Returns a report dict.

    report: {added, updated, removed, ocr, skipped_no_text, unchanged, errors}
    where ocr is a sub-count of added+updated (files recovered via OCR) and errors
    is a list of {path, error}. progress(msg) - if given - is called with human
    status lines so a long index is observable (and its crash point visible).
    """
    if ocr_fn is None:
        ocr_fn = ocr_mod.try_ocr
    effective_ocr = ocr_fn if use_ocr else None

    report = {
        "added": 0,
        "updated": 0,
        "removed": 0,
        "ocr": 0,
        "skipped_no_text": 0,
        "unchanged": 0,
        "errors": [],
    }

    disk = scan_pdfs(root)
    existing = store.file_index()

    # Orphan sweep: files in the table but gone from disk.
    for path in list(existing):
        if path not in disk:
            store.delete_file(path)
            report["removed"] += 1

    total = len(disk)
    for i, (path, file_hash) in enumerate(disk.items(), 1):
        rec = existing.get(path)
        # Unchanged only when the hash matches AND the prior pass resolved it; an
        # 'error' row is retried so transient failures self-heal.
        if rec and rec["hash"] == file_hash and rec["status"] in ("ok", "ocr", "no_text"):
            report["unchanged"] += 1
            continue

        is_new = rec is None
        if progress:
            progress(f"[{i}/{total}] {Path(path).name}")
        try:
            status = _index_one(store, path, file_hash, embed_fn, effective_ocr, progress)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the whole run
            report["errors"].append({"path": path, "error": str(exc)})
            _mark_error(store, path, file_hash)
            continue

        if status == "no_text":
            report["skipped_no_text"] += 1
            continue
        if status == "ocr":
            report["ocr"] += 1
        report["added" if is_new else "updated"] += 1

    return report


def _mark_error(store, path, file_hash):
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        mtime = None
    try:
        store.upsert_file(path, file_hash, mtime, None, "error", _now())
    except Exception:
        pass
