"""Model routing - provider selection is a pure function of configuration.

The provider comes from the corpus's 'provider:model' reference (or an explicit
--model override), never from the environment. A 'local:' corpus never reads
OPENAI_API_KEY; an 'openai:' corpus without the key is a hard setup error, not
a silent fallback.
"""

import pytest

from ask import llm


def test_local_never_reads_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
    cfg = llm.resolve_llm_config("local", "qwen2.5:3b")
    assert cfg.provider == "local"
    assert cfg.base_url == llm.DEFAULT_OLLAMA_URL
    # The resolved key is the Ollama placeholder, never the real cloud key.
    assert cfg.api_key == "ollama"
    assert cfg.api_key != "sk-should-be-ignored"
    assert cfg.model == "qwen2.5:3b"


def test_local_client_points_at_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
    cfg = llm.resolve_llm_config("local", "qwen2.5:3b")
    client, model, provider = llm.get_client_and_model(cfg)
    assert provider == "local"
    assert str(client.base_url).rstrip("/") == llm.DEFAULT_OLLAMA_URL.rstrip("/")
    assert "openai.com" not in str(client.base_url)


def test_openai_uses_key_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    cfg = llm.resolve_llm_config("openai", "gpt-5-nano")
    assert cfg.provider == "openai"
    assert cfg.api_key == "sk-real-key"
    assert cfg.base_url == llm.DEFAULT_OPENAI_URL
    assert cfg.model == "gpt-5-nano"


def test_openai_without_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMSetupError) as exc:
        llm.resolve_llm_config("openai", "gpt-5-nano")
    assert "OPENAI_API_KEY" in str(exc.value)


def test_override_to_local_wins(monkeypatch):
    # An explicit --model override replaces the corpus reference entirely.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    cfg = llm.resolve_llm_config("openai", "gpt-5-nano", override="local:llama3.1:8b")
    assert cfg.provider == "local"
    assert cfg.model == "llama3.1:8b"
    assert cfg.api_key == "ollama"


def test_override_to_openai_without_key_is_a_setup_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMSetupError):
        llm.resolve_llm_config("local", "qwen2.5:3b", override="openai:gpt-5-nano")


def test_parse_model_ref_valid():
    assert llm.parse_model_ref("openai:gpt-5-nano") == ("openai", "gpt-5-nano")
    # The provider is everything before the FIRST colon; local tags keep theirs.
    assert llm.parse_model_ref("local:qwen2.5:3b") == ("local", "qwen2.5:3b")
    assert llm.parse_model_ref("  LOCAL:qwen2.5:3b ") == ("local", "qwen2.5:3b")


@pytest.mark.parametrize(
    "bad", ["gpt-5-nano", "openai:", "local:", "", None, "ollama:qwen2.5:3b"]
)
def test_parse_model_ref_invalid(bad):
    with pytest.raises(llm.LLMSetupError):
        llm.parse_model_ref(bad)


def test_unknown_provider_raises():
    with pytest.raises(llm.LLMSetupError):
        llm.resolve_llm_config("azure", "whatever")
