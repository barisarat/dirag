"""Write the library fixture PDFs to tests/fixtures/library/ for a CLI demo.

Optional: pytest builds these in a temp dir at runtime, so this is only needed to
try `ask index library --config tests/fixtures/corpora.fixtures.toml` by hand.

    uv run python tests/fixtures/make_library_fixtures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import tests.*

from tests.pdf_utils import born_pdf, scanned_pdf

BACKUPS = (
    "Restic performs nightly encrypted backups to offsite object storage. "
    "Snapshots are pruned to keep seven daily, four weekly, and six monthly copies. "
    "A monthly restore drill restores the latest snapshot into a scratch directory. "
) * 3

SERVERS = (
    "The home server runs Arch Linux and is reachable on the local network. "
    "SSH is key-only on a high port and password authentication is disabled. "
) * 3


def main():
    out = Path(__file__).parent / "library"
    out.mkdir(parents=True, exist_ok=True)
    born_pdf(out / "backups.pdf", "Backups", BACKUPS)
    born_pdf(out / "servers.pdf", "Servers", SERVERS)
    scanned_pdf(out / "scanned.pdf", "This page is an image of words with no text layer.")
    print(f"wrote fixture PDFs to {out}")


if __name__ == "__main__":
    main()
