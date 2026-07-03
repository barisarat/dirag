"""Tiny PDF builders for the library-tier fixtures (born-digital + scanned).

Built with PyMuPDF so tests need no committed binary fixtures. born_pdf writes a
real text layer; scanned_pdf rasterizes text to an image and inserts it with NO
text layer, so parse_pdf returns nothing and the no_text / OCR path is exercised.
"""

import fitz


def born_pdf(path, title, body):
    """A born-digital PDF: text wrapped in a box so it stays on the page."""
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, body, fontsize=11)
    doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()


def scanned_pdf(path, body):
    """An image-only PDF (no text layer): text is drawn, rasterized, re-inserted."""
    tmp = fitz.open()
    tp = tmp.new_page()
    tp.insert_text((72, 100), body, fontsize=14)
    pix = tp.get_pixmap(dpi=100)
    tmp.close()

    doc = fitz.open()
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()
