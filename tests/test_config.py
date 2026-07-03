"""corpora.toml parsing, corpus validation, and default-corpus resolution."""

from pathlib import Path

import pytest

from ask import config as cfg


def _write(tmp_path, text):
    p = tmp_path / "corpora.toml"
    p.write_text(text)
    return p


def test_load_valid_corpora(tmp_path):
    path = _write(
        tmp_path,
        """
        default = "library"

        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"

        [private]
        path = "~/documents/confidential"
        retriever = "vector"
        model = "local:qwen2.5:3b"
        """,
    )
    corpora = cfg.load_corpora(path)
    assert set(corpora) == {"library", "private"}

    library = corpora["library"]
    assert library.retriever == "vector"
    assert library.provider == "openai"
    assert library.model == "gpt-5-nano"
    assert library.model_ref == "openai:gpt-5-nano"
    # Paths are expanded, not left with a literal tilde.
    assert "~" not in str(library.path)
    assert library.path == Path("~/vault/library").expanduser()

    private = corpora["private"]
    assert private.provider == "local"
    # A local model tag may itself contain colons.
    assert private.model == "qwen2.5:3b"


def test_model_is_required(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load_corpora(path)
    assert "model" in str(exc.value)


def test_invalid_model_reference(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "gpt-5-nano"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load_corpora(path)
    assert "provider:model" in str(exc.value)


def test_invalid_retriever(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "grep"
        model = "openai:gpt-5-nano"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load_corpora(path)
    assert "retriever" in str(exc.value)


def test_get_corpus_unknown_name(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.get_corpus("papers", config_path=path)
    assert "Unknown corpus" in str(exc.value)


def test_missing_config_file(tmp_path):
    with pytest.raises(cfg.ConfigError):
        cfg.load_corpora(tmp_path / "does_not_exist.toml")


def test_default_key_resolves(tmp_path):
    path = _write(
        tmp_path,
        """
        default = "papers"

        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"

        [papers]
        path = "~/vault/papers"
        retriever = "vector"
        model = "local:qwen2.5:3b"
        """,
    )
    assert cfg.resolve_corpus(config_path=path).name == "papers"
    # An explicit name always wins over the default.
    assert cfg.resolve_corpus("library", config_path=path).name == "library"


def test_single_corpus_is_implicit_default(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"
        """,
    )
    assert cfg.resolve_corpus(config_path=path).name == "library"
    corpora, default = cfg.load_config(path)
    assert default == "library"


def test_multiple_corpora_without_default_is_ambiguous(tmp_path):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"

        [papers]
        path = "~/vault/papers"
        retriever = "vector"
        model = "openai:gpt-5-nano"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve_corpus(config_path=path)
    assert "default" in str(exc.value)


def test_default_naming_unknown_corpus_fails(tmp_path):
    path = _write(
        tmp_path,
        """
        default = "nope"

        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"
        """,
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load_corpora(path)
    assert "default" in str(exc.value)


def test_env_override_search_path(tmp_path, monkeypatch):
    path = _write(
        tmp_path,
        """
        [library]
        path = "~/vault/library"
        retriever = "vector"
        model = "openai:gpt-5-nano"
        """,
    )
    monkeypatch.setenv("ASK_CORPORA", str(path))
    # No explicit config_path -> should find it via ASK_CORPORA.
    corpora = cfg.load_corpora()
    assert "library" in corpora
