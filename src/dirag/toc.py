"""A book's embedded table of contents as a page-partitioned chapter map.

Every page of a book is assigned to exactly one section, and a passage inherits
that section's title path, so a passage from deep inside a chapter still names
what it is about.

Partition rule: entries are read in TOC order, and a section owns the pages from
its own start up to the next entry that starts on a later page. A parent whose
first child begins on the same page owns no pages, but stays in its children's
title path. Pages before the first entry form a "Front matter" section, so the
spans always sum to the page count.

Broken TOCs: entries pointing outside the document are dropped, and an entry
whose page goes backwards is clamped to the previous page rather than reordered,
because TOC order is the document's reading order.

A book with no usable TOC returns no sections; the indexer then gives its
passages the book title alone as context.
"""

# A TOC is trusted only with at least this many entries reaching at least this
# far into the book; less is a front-matter stub with no chapters behind it.
MIN_TOC_ENTRIES = 5
MIN_TOC_SPAN_PCT = 50
FRONT_MATTER = "Front matter"


def raw_toc(doc):
    """TOC entries as [(depth, title, page)], dropping unusable ones."""
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        return []
    entries = []
    for item in toc:
        if len(item) < 3:
            continue
        depth, title, page = int(item[0]), (item[1] or "").strip(), int(item[2])
        # Page 0 or -1 means the entry has no resolvable destination.
        if depth < 1 or not title or page < 1 or page > doc.page_count:
            continue
        entries.append((depth, " ".join(title.split()), page))
    return entries


def has_usable_toc(doc, entries):
    """A TOC that is worth segmenting on, as opposed to a front-matter stub."""
    if len(entries) < MIN_TOC_ENTRIES or not doc.page_count:
        return False
    span_pct = 100.0 * max(page for _, _, page in entries) / doc.page_count
    return span_pct >= MIN_TOC_SPAN_PCT


def _clamped(entries):
    """Force start pages to be non-decreasing, keeping the document's own order."""
    out = []
    floor = 1
    for depth, title, page in entries:
        page = max(page, floor)
        floor = page
        out.append((depth, title, page))
    return out


def _title_path(stack, depth, title):
    """Ancestor titles for an entry, using a depth-indexed stack."""
    del stack[depth - 1:]
    while len(stack) < depth - 1:
        # A TOC that jumps 1 -> 3 leaves a hole; pad rather than misattribute.
        stack.append("")
    stack.append(title)
    return [part for part in stack if part]


def sections_of(doc):
    """Page-partitioned sections: [{depth, title, path, start_page, end_page}].

    Returns [] when the book has no usable TOC. Otherwise every page from 1 to
    page_count belongs to exactly one returned section.
    """
    entries = raw_toc(doc)
    if not has_usable_toc(doc, entries):
        return []
    entries = _clamped(entries)

    # Walk the entries building each one's title path, then keep only the last
    # entry at any given start page: it is the deepest one there, and the pages
    # that follow belong to it rather than to its parent.
    walked = []
    stack = []
    for depth, title, page in entries:
        walked.append((depth, title, _title_path(stack, depth, title), page))

    owners = []
    for index, (depth, title, path, page) in enumerate(walked):
        is_last = index + 1 == len(walked)
        if is_last or walked[index + 1][3] > page:
            owners.append((depth, title, path, page))

    sections = []
    first_page = owners[0][3]
    if first_page > 1:
        sections.append({
            "depth": 1,
            "title": FRONT_MATTER,
            "path": [FRONT_MATTER],
            "start_page": 1,
            "end_page": first_page - 1,
        })
    for index, (depth, title, path, page) in enumerate(owners):
        end = owners[index + 1][3] - 1 if index + 1 < len(owners) else doc.page_count
        sections.append({
            "depth": depth,
            "title": title,
            "path": path,
            "start_page": page,
            "end_page": max(end, page),
        })
    return sections


def chapters_of(sections):
    """Roll leaf sections up to their depth-1 ancestor: [{title, start_page, end_page, sections}].

    Leaves are in page order, so grouping by the first element of the title path
    gives contiguous chapters.
    """
    chapters = []
    for section in sections:
        top = section["path"][0] if section["path"] else section["title"]
        if chapters and chapters[-1]["title"] == top:
            chapters[-1]["end_page"] = section["end_page"]
            chapters[-1]["sections"] += 1
        else:
            chapters.append({
                "title": top,
                "start_page": section["start_page"],
                "end_page": section["end_page"],
                "sections": 1,
            })
    return chapters
