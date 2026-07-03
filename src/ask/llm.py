"""Unified LLM access for ask.

Each corpus declares its model in corpora.toml as a 'provider:model' reference;
that reference (or an explicit --model override) is the only routing input.

Routing is a pure function of configuration: the provider comes from the
corpus's `model` reference, never from the environment. Environment variables
supply credentials and endpoints only - OPENAI_API_KEY authenticates a corpus
already configured as 'openai:...', OPENAI_BASE_URL / OLLAMA_BASE_URL relocate
the endpoints. A corpus configured with a 'local:' model talks to Ollama and
never reads the API key (tests/test_routing.py covers this). A corpus
configured with an 'openai:' model requires the key and fails with a setup
error when it is missing - there is no silent fallback.
"""

import logging
import os
from dataclasses import dataclass

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
# OpenAI or any OpenAI-compatible endpoint - set OPENAI_BASE_URL for Groq/OpenRouter/etc.
# `or` (not a getenv default) so an empty-string env value still falls back to the real URL.
DEFAULT_OPENAI_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
# Generous total request timeout so slow local generations still finish; a down/refused
# Ollama raises APIConnectionError instantly regardless. Its job is to stop a hung Ollama
# (connection accepted, no response) from blocking forever.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))

VALID_PROVIDERS = ("local", "openai")


class LLMSetupError(Exception):
    """A model reference or its prerequisites are wrong (bad format, missing key).

    Raised at resolve time so the CLI can print an actionable message and exit
    before any request is attempted.
    """


@dataclass(frozen=True)
class LLMConfig:
    """A fully resolved LLM target. `provider` is 'local' or 'openai'."""

    provider: str
    base_url: str
    api_key: str
    model: str


def parse_model_ref(ref):
    """Split a 'provider:model' reference into (provider, model).

    The provider is everything before the FIRST colon, so local model tags that
    themselves contain colons work: 'local:qwen2.5:3b' -> ('local', 'qwen2.5:3b').
    Raises LLMSetupError on a malformed reference.
    """
    raw = (ref or "").strip()
    provider, sep, model = raw.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not sep or provider not in VALID_PROVIDERS or not model:
        raise LLMSetupError(
            f"Invalid model reference {ref!r}: expected 'provider:model' with a provider "
            f"in {VALID_PROVIDERS}, e.g. 'openai:gpt-5-nano' or 'local:qwen2.5:3b'."
        )
    return provider, model


def resolve_llm_config(provider, model, override=None):
    """Resolve the LLM target for a corpus.

    `provider` and `model` come from the corpus config (already split by
    parse_model_ref). `override`, when given, is a full 'provider:model'
    reference (the --model flag) and replaces both.

    - 'local'  -> the Ollama endpoint. The API key is never read on this path.
    - 'openai' -> the configured OpenAI-compatible endpoint. Requires
      OPENAI_API_KEY; raises LLMSetupError when it is missing (no fallback).
    """
    if override:
        provider, model = parse_model_ref(override)

    if provider == "local":
        return LLMConfig(
            provider="local",
            base_url=DEFAULT_OLLAMA_URL,
            # Ollama ignores the key; the OpenAI-compatible SDK requires a non-empty one.
            api_key="ollama",
            model=model,
        )

    if provider == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise LLMSetupError(
                f"This corpus is configured for 'openai:{model}' but OPENAI_API_KEY is "
                "not set. Export the key, or configure a local model instead "
                "(e.g. model = \"local:qwen2.5:3b\")."
            )
        return LLMConfig(
            provider="openai",
            base_url=DEFAULT_OPENAI_URL,
            api_key=api_key,
            model=model,
        )

    raise LLMSetupError(
        f"Unknown provider {provider!r}; expected one of {VALID_PROVIDERS}."
    )


def get_client_and_model(cfg):
    """Build an OpenAI-compatible client for a resolved LLMConfig."""
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=LLM_TIMEOUT_S)
    return client, cfg.model, cfg.provider


def list_ollama_models_status(base_url=None):
    """Probe the local Ollama for its model ids, distinguishing unreachable from empty.

    Returns {"reachable": bool, "models": [...]}. reachable=False means the request
    itself failed (Ollama down / wrong URL); reachable=True with an empty list means
    Ollama is up but has no models pulled.
    """
    base = (base_url or "").strip() or DEFAULT_OLLAMA_URL
    try:
        resp = httpx.get(f"{base.rstrip('/')}/models", timeout=1.5)
        data = resp.json()
        return {
            "reachable": True,
            "models": [m.get("id") for m in data.get("data", []) if m.get("id")],
        }
    except Exception:
        return {"reachable": False, "models": []}


def list_ollama_models(base_url=None):
    """Model ids served by the local Ollama (empty list if unreachable)."""
    return list_ollama_models_status(base_url)["models"]


def chat_create(cfg, messages, tools=None, tool_choice=None, stream=False,
                max_tokens=1000, temperature=None):
    """Single entry point for a Chat Completions call (tool-calling or plain).

    cfg is a resolved LLMConfig. tool_choice may be "auto", "none", or a forced
    {"type":"function", ...} object; defaults to "auto" when tools are supplied.
    temperature is omitted unless explicitly given, so the local model uses its
    own default.
    """
    client, model, provider = get_client_and_model(cfg)
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if provider == "openai":
        # Newer OpenAI models (gpt-5 / o-series) reject the legacy `max_tokens` and
        # require `max_completion_tokens`; they also only accept the default
        # temperature, so we omit any explicit temperature the local path would send.
        kwargs["max_completion_tokens"] = max_tokens
        # gpt-5-family reasoning models otherwise burn the whole token cap on hidden
        # reasoning (and 400 "max_tokens reached" errors on small caps). For grounded
        # extraction/JSON-verdict work minimal reasoning is right and fast. Set
        # ASK_REASONING_EFFORT="" to disable for a model that rejects the param.
        effort = (os.getenv("ASK_REASONING_EFFORT", "minimal") or "").strip()
        if effort:
            kwargs["reasoning_effort"] = effort
    else:
        # Ollama speaks the legacy Chat Completions params.
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    return client.chat.completions.create(**kwargs)


def text_of(response):
    """Plain text of a non-streamed completion."""
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError):
        return ""


def delta_of(chunk):
    """Text delta of a streamed completion chunk (empty when none)."""
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError):
        return ""


def describe_llm_error(cfg, exc):
    """Turn an LLM/generation exception into an actionable, user-facing signal.

    Returns {"code", "message", "retryable"}. Anything that isn't a recognised
    LLM/network failure falls back to a generic message.
    """
    base_url, model = cfg.base_url, cfg.model
    is_openai = cfg.provider == "openai"
    engine_label = "the cloud model" if is_openai else "the local model"

    # OpenAI-specific failures are APIStatusError subclasses, so they MUST be checked
    # before the generic APIStatusError/OpenAIError fallthrough below.
    if isinstance(exc, AuthenticationError):
        return {
            "code": "openai_auth",
            "retryable": False,
            "message": "Your API key was rejected (401). Check or replace OPENAI_API_KEY.",
        }
    if isinstance(exc, PermissionDeniedError):
        return {
            "code": "openai_forbidden",
            "retryable": False,
            "message": (
                f"Your key can't access the model '{model}' (403). Choose a model your "
                "account can use."
            ),
        }
    if isinstance(exc, RateLimitError):
        out_of_quota = "quota" in (str(getattr(exc, "message", "")) or str(exc)).lower()
        if out_of_quota:
            return {
                "code": "openai_quota",
                "retryable": False,
                "message": "Your account is out of quota/credits. Add billing, or use a local model.",
            }
        return {
            "code": "openai_rate_limit",
            "retryable": True,
            "message": "Cloud rate limit hit (429). Wait a moment and try again.",
        }
    if isinstance(exc, APITimeoutError):
        return {
            "code": "timeout",
            "retryable": True,
            "message": (
                f"{engine_label} took too long to respond (>{int(LLM_TIMEOUT_S)}s). "
                "Try again, or pick a smaller/faster model."
            ),
        }
    if isinstance(exc, APIConnectionError):
        if is_openai:
            return {
                "code": "openai_unreachable",
                "retryable": True,
                "message": (
                    f"Couldn't reach the cloud endpoint at {base_url}. Check your network "
                    "(or OPENAI_BASE_URL), then try again."
                ),
            }
        return {
            "code": "ollama_unreachable",
            "retryable": True,
            "message": (
                f"Couldn't reach your local Ollama at {base_url}. Make sure Ollama is "
                "running on the host, then try again."
            ),
        }
    if isinstance(exc, NotFoundError) or (
        isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 404
    ):
        if is_openai:
            return {
                "code": "model_missing",
                "retryable": False,
                "message": f"The model '{model}' isn't available for your account.",
            }
        return {
            "code": "model_missing",
            "retryable": False,
            "message": (
                f"The model '{model}' isn't available in Ollama. Pull it with "
                f"`ollama pull {model}`, or choose another model."
            ),
        }
    if isinstance(exc, (APIStatusError, OpenAIError)):
        status = getattr(exc, "status_code", None)
        suffix = f" (status {status})" if status else ""
        return {
            "code": "llm_error",
            "retryable": True,
            "message": f"{engine_label} returned an error{suffix}. Please try again.",
        }
    return {
        "code": "internal",
        "retryable": True,
        "message": "Something went wrong generating that answer. Please try again.",
    }
