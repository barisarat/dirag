"""Shared answer pipeline: ranking, citation cleaning, revision guard (no LLM)."""

from ask import answer


def _src(url, text, score=1.0, line=1):
    """A line-addressed source (no page/bbox geometry)."""
    return {
        "url": url,
        "title": url.split("/")[-1],
        "text": text,
        "score": score,
        "chunkId": f"{url}:{line}",
        "metadata": {"line": line},
    }


def test_select_answer_context_prefers_lexically_covering_source():
    sources = [
        _src("a.md", "the cat sat on the mat", line=1),
        _src("b.md", "backups are pruned to keep daily weekly monthly copies", line=2),
    ]
    selected = answer.select_answer_context("how are backups pruned", sources, max_items=6)
    # The covering source (b.md) should be selected and ranked ahead of the noise.
    assert selected[0]["url"] == "b.md"


def test_select_answer_context_empty_results():
    assert answer.select_answer_context("q", [], max_items=6) == []


def test_select_answer_context_no_query_tokens_returns_prefix():
    sources = [_src("a.md", "x", line=1), _src("b.md", "y", line=2)]
    # A query of only stopwords tokenizes to nothing -> fall back to prefix.
    selected = answer.select_answer_context("the a of", sources, max_items=1)
    assert len(selected) == 1


def test_make_citation_line_addressed_shape():
    c = answer.make_citation(_src("notes/servers.md", "SSH is key-only on port 2222", line=7))
    assert c["label"] == "servers.md"
    assert c["ref"] == "notes/servers.md"
    assert c["line"] == 7
    assert c["page"] is None
    assert "2222" in c["passage"]


def test_remove_invalid_citation_markers():
    # [3] is out of range for a 2-citation answer and must be stripped.
    cleaned = answer.remove_invalid_citation_markers("Foo [1] bar [3] baz [2].", citation_count=2)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[3]" not in cleaned


def test_accept_revision_keeps_citations():
    original = "The rate is 90 EUR [1] with net 14 terms [2]."
    good = "The rate is 90 EUR per hour [1], net 14 [2]."
    assert answer.accept_revision(original, good) is True


def test_accept_revision_rejects_dropped_citations():
    original = "A [1] B [2] C [3]."
    stripped = "A plain sentence with no markers."
    assert answer.accept_revision(original, stripped) is False


def test_accept_revision_rejects_truncation():
    original = "A fairly long grounded answer [1] that says several things about the topic."
    tiny = "A [1]."
    assert answer.accept_revision(original, tiny) is False


def test_accept_revision_rejects_empty():
    assert answer.accept_revision("A [1] B [2].", "") is False


def test_format_context_numbers_sources_with_locator():
    citations = [
        answer.make_citation(_src("a.md", "alpha content", line=3)),
        answer.make_citation(_src("b.md", "beta content", line=9)),
    ]
    block = answer.format_context(citations)
    assert "[1] a.md (line 3)" in block
    assert "[2] b.md (line 9)" in block
    assert "alpha content" in block
