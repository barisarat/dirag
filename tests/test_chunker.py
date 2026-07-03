"""PDF chunking: passage windowing, overlap, union bbox, batching (no LLM, no PDF)."""

from ask.library import chunker


def _block(text, bbox):
    return {"text": text, "bbox": bbox}


def test_page_passages_single_short_block():
    passages = chunker.page_passages([_block("a short line", [0, 0, 10, 10])])
    assert len(passages) == 1
    assert passages[0]["text"] == "a short line"
    assert passages[0]["bbox"] == [0, 0, 10, 10]


def test_page_passages_splits_when_over_target():
    # Three ~400-char blocks: 400+400 fits one buffer, the third forces a flush.
    a = "a" * 400
    b = "b" * 400
    c = "c" * 400
    blocks = [
        _block(a, [0, 0, 10, 10]),
        _block(b, [0, 10, 10, 20]),
        _block(c, [0, 20, 10, 30]),
    ]
    passages = chunker.page_passages(blocks)
    assert len(passages) >= 2
    # Every passage stays within a sane bound of the target window.
    for p in passages:
        assert len(p["text"]) <= chunker.TARGET_CHARS + chunker.OVERLAP_CHARS + 2


def test_page_passages_overlap_carries_tail():
    a = "a" * 500
    b = "b" * 500
    passages = chunker.page_passages([_block(a, [0, 0, 1, 1]), _block(b, [0, 1, 1, 2])])
    assert len(passages) == 2
    tail = passages[0]["text"][-chunker.OVERLAP_CHARS:]
    # The second passage begins with the carried-over tail of the first.
    assert passages[1]["text"].startswith(tail)


def test_page_passages_union_bbox():
    a = "a" * 500
    b = "b" * 500
    # Two blocks that fit in one buffer -> one passage whose bbox spans both.
    passages = chunker.page_passages([
        _block("short a", [0, 0, 5, 5]),
        _block("short b", [10, 10, 20, 20]),
    ])
    assert len(passages) == 1
    assert passages[0]["bbox"] == [0, 0, 20, 20]


def test_page_passages_skips_empty_blocks():
    passages = chunker.page_passages([_block("   ", [0, 0, 1, 1]), _block("real", [0, 1, 2, 2])])
    assert len(passages) == 1
    assert passages[0]["text"] == "real"


def test_union_bbox_none_for_empty():
    assert chunker.union_bbox([]) is None


def test_batched():
    assert list(chunker.batched(list(range(10)), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert list(chunker.batched([], 3)) == []
