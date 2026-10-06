"""Where dirag keeps its files, and the one library folder it serves.

Two directories, both overridable:

    DIRAG_HOME    derived data: one index per library folder (indexes/), the
                  embedding model cache (models/) and the indexing job record
                  (job.json). Default ~/.local/share/dirag. Rebuildable.
    DIRAG_STATE   user state: config.json, positions.json, bookmarks.json,
                  cards.json. Default DIRAG_HOME. Small, textual, not derived.

The library is a single folder of PDFs. It is resolved in this order: the
--library option, the DIRAG_LIBRARY variable, then config.json. Passing
--library, or choosing a folder in the app, saves it to config.json. With
DIRAG_LIBRARY set the folder is fixed and the app cannot change it.

The app's folder picker lists folders under DIRAG_BROWSE_ROOT only (default:
the home directory).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

HOME = Path(os.getenv("DIRAG_HOME") or Path.home() / ".local" / "share" / "dirag").expanduser()
STATE = Path(os.getenv("DIRAG_STATE") or HOME).expanduser()
MODELS = HOME / "models"
JOB = HOME / "job.json"
CONFIG = STATE / "config.json"
FIXED = bool(os.getenv("DIRAG_LIBRARY"))
BROWSE_ROOT = Path(os.getenv("DIRAG_BROWSE_ROOT") or Path.home()).expanduser().resolve()


def read_json(path, empty):
    """The file's content, or `empty` when it is missing, unreadable or the wrong type."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, type(empty)) else empty
    except FileNotFoundError:
        return empty
    except (OSError, ValueError) as exc:
        print(f"{Path(path).name} unusable ({exc}); ignoring it", file=sys.stderr)
        return empty


def write_json(path, data):
    """Write through a temp file and rename, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def index_path(root):
    """The index file for a library folder: its name plus a hash of its absolute path."""
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:10]
    return HOME / "indexes" / f"{root.name or 'root'}-{digest}.sqlite3"


def set_library(path):
    """Save `path` as the library folder and return it as an absolute Path."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")
    settings = read_json(CONFIG, {})
    settings["library"] = str(root)
    write_json(CONFIG, settings)
    return root


def library(override=None):
    """The library folder as an absolute Path, or None when none is set."""
    if override:
        return set_library(override)
    value = os.getenv("DIRAG_LIBRARY") or read_json(CONFIG, {}).get("library")
    return Path(value).expanduser().resolve() if value else None


def require_library(override=None):
    root = library(override)
    if root is None:
        raise SystemExit("No library folder set. Pass --library PATH once; it is remembered.")
    return root
