"""PDF -> ordered text blocks with page + bbox, via PyMuPDF (fitz).

The swappable parsing interface: parse_pdf(path) returns blocks in human reading
order, each {text, page, bbox} (page 1-based; bbox = [x0, y0, x1, y1] in PDF
points). Two-column reading order - where a naive extractor reads straight across
both columns - is fixed by sorting blocks into a left/right column per page.
Single-column books degrade fine. This module can be swapped for another
extractor without touching the chunker or store.
"""

import fitz  # PyMuPDF


def _column_index(block, mid):
    """0 for the left column, 1 for the right, by the block's horizontal center.
    Full-width blocks (title/abstract) center near the page middle and fall into
    the left stream, which reads correctly for the top-of-page spanning content."""
    x0, x1 = block[0], block[2]
    center = (x0 + x1) / 2.0
    return 0 if center < mid else 1


def _ordered_blocks(page):
    raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [
        b for b in raw
        if len(b) >= 5 and (b[4] or "").strip() and (len(b) <= 6 or b[6] == 0)
    ]

    mid = page.rect.width / 2.0
    # Read left column top-to-bottom, then right column - within a column, by y then x.
    text_blocks.sort(key=lambda b: (_column_index(b, mid), round(b[1], 1), b[0]))

    return [
        {"text": " ".join(b[4].split()), "bbox": [b[0], b[1], b[2], b[3]]}
        for b in text_blocks
    ]


def parse_pdf(path):
    """Return [{text, page, bbox}] in reading order across all pages."""
    doc = fitz.open(path)
    blocks = []

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for blk in _ordered_blocks(page):
                if not blk["text"]:
                    continue
                blocks.append({
                    "text": blk["text"],
                    "page": page_index + 1,  # 1-based for citations
                    "bbox": blk["bbox"],
                })
    finally:
        doc.close()

    return blocks


def page_count(path):
    """Number of pages in the PDF (for the files table). 0 on any open failure."""
    try:
        doc = fitz.open(path)
    except Exception:
        return 0
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def pdf_title(path):
    """Best-effort document title: PDF metadata title if present, else None."""
    doc = fitz.open(path)
    try:
        title = (doc.metadata or {}).get("title") or ""
    finally:
        doc.close()
    return title.strip() or None
