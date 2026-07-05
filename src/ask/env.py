"""Optional .env autoload for credentials and endpoints.

The environment supplies credentials and endpoints (OPENAI_API_KEY,
OPENAI_BASE_URL, OLLAMA_BASE_URL); routing is still a pure function of config.
This loader only makes that env easier to populate: instead of exporting keys by
hand every session, drop them in a .env file and the CLI reads them at startup.

Search order (first match wins per key, and a real exported variable always wins
over any file - so `export OPENAI_API_KEY=...` still overrides the file):

    1. $ASK_ENV        explicit path
    2. ./.env          current directory
    3. ~/.config/ask/.env

Only KEY=VALUE lines are read: blanks and `#` comments are skipped, an optional
leading `export ` is allowed, and surrounding single/double quotes are stripped.
A malformed line is ignored, never fatal - a bad .env must not break the CLI.
"""

import os
from pathlib import Path


def _candidate_paths():
    env_path = (os.getenv("ASK_ENV") or "").strip()
    if env_path:
        yield Path(env_path).expanduser()
    yield Path(".env")
    yield Path("~/.config/ask/.env").expanduser()


def _parse(text):
    """Yield (key, value) pairs from .env text; skip blanks, comments, junk."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        yield key, value


def load_env_files():
    """Populate os.environ from the first .env files found, without overriding.

    A variable already present in the environment is left untouched, so an
    explicit `export` always wins over the file. Returns the list of files read.
    """
    loaded = []
    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for key, value in _parse(text):
            os.environ.setdefault(key, value)
        loaded.append(str(path))
    return loaded
