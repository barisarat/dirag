"""`.env` autoload - populate credentials/endpoints from a file, without override.

The loader only makes the environment easier to fill: it never overrides a
variable that is already exported (an explicit `export` wins), and a malformed
line is ignored rather than fatal. It reads $ASK_ENV, then ./.env, then
~/.config/ask/.env, first value per key winning.
"""

import os

from ask import env


def _isolate(tmp_path, monkeypatch):
    """Run in an empty cwd + HOME so only files this test writes are reachable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ASK_ENV", raising=False)


def test_reads_key_value_with_quotes_and_export(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        '# a comment\nexport OPENAI_API_KEY = "sk-fromfile"\n', encoding="utf-8"
    )
    env.load_env_files()
    assert os.environ["OPENAI_API_KEY"] == "sk-fromfile"


def test_exported_value_wins_over_file(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-fromfile\n", encoding="utf-8")
    env.load_env_files()
    assert os.environ["OPENAI_API_KEY"] == "sk-exported"


def test_malformed_lines_are_ignored(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "NO EQUALS HERE\n=missing-key\nOPENAI_BASE_URL=https://x/v1\n", encoding="utf-8"
    )
    env.load_env_files()
    assert os.environ["OPENAI_BASE_URL"] == "https://x/v1"


def test_explicit_ask_env_path_is_read(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    custom = tmp_path / "secrets.env"
    custom.write_text("OPENAI_API_KEY=sk-explicit\n", encoding="utf-8")
    monkeypatch.setenv("ASK_ENV", str(custom))
    env.load_env_files()
    assert os.environ["OPENAI_API_KEY"] == "sk-explicit"


def test_missing_files_are_a_noop(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    # No .env anywhere reachable: returns cleanly with nothing loaded.
    assert env.load_env_files() == []
