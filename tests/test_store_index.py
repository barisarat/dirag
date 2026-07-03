"""Store + incremental index: add, no-op re-run, change, orphan sweep, OCR/no_text.

Deterministic and offline: a fake embedder (no fastembed) and, for the OCR case,
a fake OCR that yields a born-digital PDF. Requires sqlite-vec and pymupdf.
"""

import tempfile

from ask.library import index as index_mod
from ask.library.store import Store

from tests.pdf_utils import born_pdf


def _store(tmp_path):
    return Store(tmp_path / "data" / "library.sqlite3")


def test_index_adds_and_reports(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)
    report = index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    assert report["added"] == 2          # two born-digital PDFs
    assert report["skipped_no_text"] == 1  # the scanned one, no OCR available
    assert report["updated"] == 0
    assert report["removed"] == 0

    c = store.counts()
    assert c["files"] == 3
    assert c["by_status"].get("ok") == 2
    assert c["by_status"].get("no_text") == 1
    assert c["chunks"] >= 2
    # Every chunk has exactly one vector - no orphaned vectors.
    assert c["vectors"] == c["chunks"]
    store.close()


def test_reindex_is_noop(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)
    index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)
    before = store.counts()

    report = index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)
    assert report["added"] == 0
    assert report["updated"] == 0
    assert report["removed"] == 0
    assert report["unchanged"] == 3  # 2 ok + 1 no_text all recognized as unchanged

    after = store.counts()
    assert after["chunks"] == before["chunks"]
    assert after["vectors"] == before["vectors"]
    store.close()


def test_changed_file_is_reindexed(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)
    index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    # Rewrite one book with new content -> new hash -> update, old chunks swept.
    born_pdf(library_dir / "backups.pdf", "Backups v2", "Completely different body text here. " * 20)
    report = index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    assert report["updated"] == 1
    assert report["added"] == 0
    assert report["unchanged"] == 2  # servers.pdf + scanned.pdf

    c = store.counts()
    # No orphaned vectors after the swap.
    assert c["vectors"] == c["chunks"]
    store.close()


def test_orphan_sweep_on_delete(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)
    index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    (library_dir / "servers.pdf").unlink()
    report = index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    assert report["removed"] == 1
    c = store.counts()
    assert c["files"] == 2  # backups.pdf (ok) + scanned.pdf (no_text)
    # Deleted file's chunks are gone; vectors still match chunks.
    assert c["vectors"] == c["chunks"]
    remaining = set(store.file_index())
    assert not any(p.endswith("servers.pdf") for p in remaining)
    store.close()


def test_ocr_fallback_marks_ocr_status(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)

    def fake_ocr(src_path):
        # Simulate ocrmypdf producing a searchable copy of the scanned page.
        tmp = tempfile.NamedTemporaryFile(prefix="ask_ocr_", suffix=".pdf", delete=False)
        tmp.close()
        born_pdf(tmp.name, "OCR", "Recovered text from the scanned page now selectable. " * 8)
        return tmp.name

    report = index_mod.build_index(
        store, library_dir, embed_fn=fake_embed, use_ocr=True, ocr_fn=fake_ocr
    )
    assert report["skipped_no_text"] == 0
    assert report["ocr"] == 1
    assert report["added"] == 3  # both born-digital + the OCR-recovered scanned one

    c = store.counts()
    assert c["by_status"].get("ocr") == 1
    assert c["by_status"].get("ok") == 2
    assert c["vectors"] == c["chunks"]
    store.close()


def test_search_returns_nearest_chunk(tmp_path, library_dir, fake_embed):
    store = _store(tmp_path)
    index_mod.build_index(store, library_dir, embed_fn=fake_embed, use_ocr=False)

    # Query with one of the deterministic fake vectors; KNN must return rows.
    from ask.library.embedder import EMBED_DIM

    hits = store.search([1.0] * EMBED_DIM, k=3)
    assert 1 <= len(hits) <= 3
    assert "text" in hits[0] and "path" in hits[0] and "distance" in hits[0]
    store.close()
