"""Config resolution for ask: corpora.toml -> corpus objects.

corpora.toml declares one section per corpus plus an optional top-level
`default` key naming the corpus used when none is given on the command line.
Each corpus section requires:

  path      the folder of PDFs (scanned recursively by `ask index`)
  retriever "vector" (sqlite-vec KNN over local embeddings)
  model     "provider:model" - the model that answers questions for this
            corpus, e.g. "openai:gpt-5-nano" or "local:qwen2.5:3b"

This module only resolves WHERE the config lives and validates its shape. It
does not construct any LLM client; llm.resolve_llm_config is fed the corpus's
provider/model from here.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMSetupError, parse_model_ref

VALID_RETRIEVERS = ("vector",)


class ConfigError(Exception):
    """Raised when corpora.toml is missing, malformed, or names an unknown corpus."""


@dataclass(frozen=True)
class Corpus:
    """One corpus section from corpora.toml, validated and path-expanded."""

    name: str
    path: Path
    retriever: str        # 'vector'
    provider: str         # 'local' | 'openai' (from the model reference)
    model: str            # model id (the part after 'provider:')

    @property
    def model_ref(self):
        return f"{self.provider}:{self.model}"


def _candidate_paths():
    """Config search order: $ASK_CORPORA, ./corpora.toml, ~/.config/ask/corpora.toml."""
    env_path = (os.getenv("ASK_CORPORA") or "").strip()
    if env_path:
        yield Path(env_path).expanduser()
    yield Path("corpora.toml")
    yield Path("~/.config/ask/corpora.toml").expanduser()


def find_config_path():
    """First existing config path, or None if no config file is found."""
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    return None


def _load(config_path=None):
    """Parse corpora.toml -> ({name: Corpus}, default_name_or_None)."""
    path = Path(config_path).expanduser() if config_path else find_config_path()
    if path is None:
        raise ConfigError(
            "No corpora.toml found. Looked at $ASK_CORPORA, ./corpora.toml, and "
            "~/.config/ask/corpora.toml. Copy the example config and edit its paths."
        )
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc

    corpora = {}
    for name, section in raw.items():
        if not isinstance(section, dict):
            # Top-level scalars (like `default`) are not corpus definitions.
            continue
        corpora[name] = _build_corpus(name, section, path)

    if not corpora:
        raise ConfigError(f"No corpora defined in {path}.")

    default = raw.get("default")
    if default is not None:
        default = str(default).strip()
        if default not in corpora:
            known = ", ".join(sorted(corpora))
            raise ConfigError(
                f"default = {default!r} in {path} does not name a defined corpus. "
                f"Defined corpora: {known}."
            )

    return corpora, default


def _build_corpus(name, section, config_path):
    missing = [key for key in ("path", "retriever", "model") if key not in section]
    if missing:
        raise ConfigError(
            f"Corpus [{name}] in {config_path} is missing required key(s): "
            f"{', '.join(missing)}."
        )

    retriever = str(section["retriever"]).strip().lower()
    if retriever not in VALID_RETRIEVERS:
        raise ConfigError(
            f"Corpus [{name}] has retriever={retriever!r}; expected one of "
            f"{VALID_RETRIEVERS}."
        )

    try:
        provider, model = parse_model_ref(str(section["model"]))
    except LLMSetupError as exc:
        raise ConfigError(f"Corpus [{name}] in {config_path}: {exc}") from exc

    path = Path(str(section["path"])).expanduser()
    return Corpus(
        name=name, path=path, retriever=retriever, provider=provider, model=model
    )


def load_config(config_path=None):
    """Parse corpora.toml -> ({name: Corpus}, default_corpus_name_or_None).

    When no `default` key is set and exactly one corpus is defined, that corpus
    is returned as the default.
    """
    corpora, default = _load(config_path)
    if default is None and len(corpora) == 1:
        default = next(iter(corpora))
    return corpora, default


def load_corpora(config_path=None):
    """Parse corpora.toml into {name: Corpus}. Raises ConfigError on any problem."""
    corpora, _default = _load(config_path)
    return corpora


def get_corpus(name, config_path=None):
    """Resolve a single named corpus. Raises ConfigError if it is not defined."""
    corpora, _default = _load(config_path)
    if name not in corpora:
        known = ", ".join(sorted(corpora)) or "(none)"
        raise ConfigError(f"Unknown corpus {name!r}. Defined corpora: {known}.")
    return corpora[name]


def resolve_corpus(name=None, config_path=None):
    """Resolve the corpus to use for a command.

    An explicit `name` wins. Otherwise the top-level `default` from corpora.toml
    is used; when no default is set and exactly one corpus is defined, that
    corpus is the default. Anything else is ambiguous and raises ConfigError.
    """
    corpora, default = _load(config_path)

    if name:
        if name not in corpora:
            known = ", ".join(sorted(corpora)) or "(none)"
            raise ConfigError(f"Unknown corpus {name!r}. Defined corpora: {known}.")
        return corpora[name]

    if default:
        return corpora[default]

    if len(corpora) == 1:
        return next(iter(corpora.values()))

    known = ", ".join(sorted(corpora))
    raise ConfigError(
        f"Multiple corpora are defined ({known}) and no default is set. Pass "
        "--corpus <name>, or add `default = \"<name>\"` at the top of corpora.toml."
    )
