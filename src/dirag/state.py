"""User state in the state directory, keyed by a book's path relative to the library.

    positions.json   {path: page}   where reading stopped in each book
    bookmarks.json   [path, ...]    the books on the Home view
    cards.json       {path: card}   optional display overrides: title, subtitle,
                                    authors (list), year, pages

Keys are paths, not index ids, so state survives a rebuild of the index.

Every write changes one key and is merged into the file on disk. A client holds
a copy from page load, so writing a whole document from a stale tab would undo
changes made since in another tab.
"""

from . import config

POSITIONS = config.STATE / "positions.json"
BOOKMARKS = config.STATE / "bookmarks.json"
CARDS = config.STATE / "cards.json"


def positions():
    return config.read_json(POSITIONS, {})


def set_position(path, page):
    data = positions()
    data[path] = int(page)
    config.write_json(POSITIONS, data)


def bookmarks():
    return [p for p in config.read_json(BOOKMARKS, []) if isinstance(p, str)]


def set_bookmark(path, on):
    paths = bookmarks()
    paths = (paths + [path] if path not in paths else paths) if on else [p for p in paths if p != path]
    config.write_json(BOOKMARKS, paths)
    return paths


def cards():
    """Read on every call, so an edited cards.json shows on reload without a restart."""
    return config.read_json(CARDS, {})
