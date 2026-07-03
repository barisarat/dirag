"""PDF chunking: window per-page blocks into sub-page passages (page + bbox).

page_passages windows a page's text blocks into overlapping ~TARGET_CHARS spans,
each tagged with the union bbox of its contributing blocks so a citation points
at a tight region. union_bbox and batched are small shared helpers.
"""

# Passage sizing: sub-page windows so a citation points at a tight span, with a
# little overlap so a sentence split across the window boundary stays retrievable.
TARGET_CHARS = 900
OVERLAP_CHARS = 150


def union_bbox(bboxes):
    if not bboxes:
        return None
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def page_passages(blocks):
    """Window a single page's blocks into ~TARGET_CHARS passages with overlap.
    Returns [{text, bbox}] where bbox is the union of the contributing blocks."""
    passages = []
    buf_text = ""
    buf_bboxes = []

    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue

        if buf_text and len(buf_text) + 1 + len(text) > TARGET_CHARS:
            passages.append({"text": buf_text, "bbox": union_bbox(buf_bboxes)})
            tail = buf_text[-OVERLAP_CHARS:]
            buf_text = f"{tail} {text}".strip()
            buf_bboxes = [block["bbox"]]
        else:
            buf_text = f"{buf_text} {text}".strip() if buf_text else text
            buf_bboxes.append(block["bbox"])

    if buf_text:
        passages.append({"text": buf_text, "bbox": union_bbox(buf_bboxes)})

    return passages


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]
