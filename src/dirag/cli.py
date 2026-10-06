"""Command line: `dirag [serve]`, `dirag index`, `dirag find`, `dirag where`, `dirag toc`."""

import argparse
import shutil
import sys
import textwrap

from . import __version__, config


def cmd_serve(args):
    from .server import serve
    if args.library:
        config.set_library(args.library)
    return serve(args.host, args.port, browser=not args.no_browser)


def cmd_index(args):
    from .index import run
    root = config.require_library(args.library)
    action = "reindex" if args.reindex else "rechunk" if args.rechunk else "update"
    return run(root, config.index_path(root), action=action, limit=args.limit)


def _index():
    return config.index_path(config.require_library())


def cmd_find(args):
    from . import search
    conn = search.connect(_index())
    try:
        rows = search.passages(conn, args.query, args.mode, args.rerank, limit=args.k, per_book=args.per_book)
    finally:
        conn.close()
    if not rows:
        print("No matches.")
        return 0
    width = min(shutil.get_terminal_size((100, 24)).columns, 100)
    for number, row in enumerate(rows, 1):
        print(f"{number:>3}  {row['book'][:66]}   p.{row['page']}{'   (dropped by llm)' if row.get('dropped') else ''}")
        if row["section"]:
            print(f"     {row['section'][:90]}")
        print(textwrap.fill(row["text"], width=width, initial_indent="     ", subsequent_indent="     "))
        print("")
    return 0


def cmd_where(args):
    from . import search
    conn = search.connect(_index())
    try:
        ranked = search.chapters(conn, search.candidates(conn, args.query, args.mode), limit=args.k)
    finally:
        conn.close()
    if not ranked:
        print("No matches.")
        return 0
    print(f"{'hits':>5}  {'pages':>11}  book / chapter")
    for entry in ranked:
        span = f"{min(entry['pages'])}-{max(entry['pages'])}"
        stamp = f" ({entry['year']})" if entry["year"] else ""
        print(f"{entry['hits']:>5}  {span:>11}  {entry['book'][:60]}{stamp}")
        print(f"{'':>5}  {'':>11}  {entry['chapter'][:70]}")
    return 0


def cmd_toc(args):
    import pymupdf
    from .toc import chapters_of, sections_of
    doc = pymupdf.open(args.pdf)
    try:
        sections = sections_of(doc)
        if not sections:
            print("No usable table of contents.")
            return 0
        for chapter in chapters_of(sections):
            print(f"p.{chapter['start_page']:>5}-{chapter['end_page']:>5}  {chapter['sections']:>4} sec  {chapter['title'][:80]}")
    finally:
        doc.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dirag", description="Search a folder of PDF books and read the page.")
    parser.add_argument("--version", action="version", version=f"dirag {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the web app (the default)")
    serve.add_argument("--library", help="the folder of PDFs; remembered once given")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8008)
    serve.add_argument("--no-browser", action="store_true", help="do not open the browser")
    serve.set_defaults(func=cmd_serve)

    index = sub.add_parser("index", help="index new and changed PDFs, drop deleted ones")
    index.add_argument("--library", help="the folder of PDFs; remembered once given")
    index.add_argument("--limit", type=int, default=0, help="only the first N PDFs")
    mode = index.add_mutually_exclusive_group()
    mode.add_argument("--reindex", action="store_true", help="parse and embed every PDF again")
    mode.add_argument("--rechunk", action="store_true", help="rebuild passages and vectors from cached pages")
    index.set_defaults(func=cmd_index)

    find = sub.add_parser("find", help="passages for a query")
    find.add_argument("query")
    find.add_argument("-k", type=int, default=10)
    find.add_argument("--per-book", type=int, default=2)
    find.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    find.add_argument("--rerank", choices=("off", "neural", "llm"), default="off")
    find.set_defaults(func=cmd_find)

    where = sub.add_parser("where", help="chapters for a query")
    where.add_argument("query")
    where.add_argument("-k", type=int, default=10)
    where.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    where.set_defaults(func=cmd_where)

    toc = sub.add_parser("toc", help="the chapter map of one PDF")
    toc.add_argument("pdf")
    toc.set_defaults(func=cmd_toc)

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in sub.choices and argv[0] not in ("-h", "--help", "--version")):
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    return args.func(args)
