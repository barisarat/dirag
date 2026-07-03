"""Shared fixtures for the library-tier tests."""

import pytest

from tests.pdf_utils import born_pdf, scanned_pdf

BACKUPS_BODY = (
    "Restic performs nightly encrypted backups to offsite object storage. "
    "Snapshots are pruned to keep seven daily, four weekly, and six monthly copies. "
    "A monthly restore drill restores the latest snapshot into a scratch directory "
    "and diffs it against a known checksum manifest to prove the backup is usable. "
) * 3

SERVERS_BODY = (
    "The home server runs Arch Linux and is reachable on the local network. "
    "SSH is key-only on a high port and password authentication is disabled. "
    "A UPS provides several minutes of runtime and triggers a clean shutdown. "
) * 3


@pytest.fixture
def library_dir(tmp_path):
    """A corpus dir: two born-digital PDFs and one scanned (no text layer)."""
    d = tmp_path / "library"
    d.mkdir()
    born_pdf(d / "backups.pdf", "Backups", BACKUPS_BODY)
    born_pdf(d / "servers.pdf", "Servers", SERVERS_BODY)
    scanned_pdf(d / "scanned.pdf", "This page is an image of words with no text layer.")
    return d


@pytest.fixture
def fake_embed():
    """Deterministic 384-dim embedder - no fastembed, no model download."""
    from ask.library.embedder import EMBED_DIM

    def _embed(texts):
        texts = list(texts)
        return [[float((i % 5) + 1)] * EMBED_DIM for i in range(len(texts))]

    return _embed
