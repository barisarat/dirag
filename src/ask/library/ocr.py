"""Optional OCR fallback for scanned PDFs with no text layer, via ocrmypdf.

ocrmypdf + tesseract are host tools (not uv deps). When a PDF parses to little or
no text, the indexer calls try_ocr: if ocrmypdf is available it produces a
searchable copy that the parser can then read (status 'ocr'); if it is not
available, try_ocr returns None and the indexer records status 'no_text' and
REPORTS it - a scanned book is never silently dropped.
"""

import shutil
import subprocess
import tempfile

OCRMYPDF_BIN = "ocrmypdf"


def ocr_available():
    return shutil.which(OCRMYPDF_BIN) is not None


def try_ocr(src_path):
    """OCR src_path to a temporary searchable PDF; return its path, or None.

    Returns None when ocrmypdf is unavailable or the run fails. The caller owns
    the returned temp file and is responsible for deleting it after re-parsing.
    """
    if not ocr_available():
        return None

    tmp = tempfile.NamedTemporaryFile(prefix="ask_ocr_", suffix=".pdf", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        OCRMYPDF_BIN,
        "--force-ocr",   # rasterize+OCR even if a broken/empty text layer exists
        "--quiet",
        str(src_path),
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        _safe_unlink(out_path)
        return None

    if proc.returncode == 0:
        return out_path

    _safe_unlink(out_path)
    return None


def _safe_unlink(path):
    try:
        import os

        os.unlink(path)
    except OSError:
        pass
