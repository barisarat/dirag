"""The language model, served by Ollama, for the LLM rerank and the AI answer.

Both are available when Ollama answers and has the model; otherwise they are
hidden and the rest of dirag works as usual. Availability is checked live
(cached for a few seconds), so starting Ollama later turns them on.

    DIRAG_LLM_MODEL   default qwen2.5:7b-instruct
    DIRAG_LLM_URL     default http://127.0.0.1:11434 (Ollama on this machine)

Every request sets an 8k context window, which the prompts need.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = os.getenv("DIRAG_LLM_MODEL") or "qwen2.5:7b-instruct"
URL = (os.getenv("DIRAG_LLM_URL") or "http://127.0.0.1:11434").rstrip("/")
CONTEXT = 8192
TIMEOUT = 180
CHECK_EVERY = 5

_checked = (0.0, None)
_pulling = threading.Event()


class LLMError(RuntimeError):
    pass


def _get(path, timeout=2):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def status():
    """'ready', 'missing' (Ollama up, model not pulled), 'pulling', or 'off' (Ollama not reachable)."""
    global _checked
    at, value = _checked
    if time.monotonic() - at < CHECK_EVERY and value is not None:
        return value
    try:
        names = {m.get("name") for m in _get("/api/tags").get("models", [])}
        value = "ready" if MODEL in names or f"{MODEL}:latest" in names else "pulling" if _pulling.is_set() else "missing"
    except (urllib.error.URLError, OSError, ValueError):
        value = "off"
    _checked = (time.monotonic(), value)
    return value


def enabled():
    return status() == "ready"


def pull_in_background():
    """Pull the model when Ollama is up without it, printing progress to the terminal."""
    if status() != "missing" or _pulling.is_set():
        return
    _pulling.set()
    threading.Thread(target=_pull, daemon=True).start()


def _pull():
    global _checked
    request = urllib.request.Request(f"{URL}/api/pull", data=json.dumps({"model": MODEL}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            for line in response:
                event = json.loads(line or b"{}")
                if event.get("error"):
                    raise LLMError(event["error"])
                done, total = event.get("completed"), event.get("total")
                share = f" {100 * done / total:.0f}%" if done and total else ""
                print(f"\r  AI    pulling {MODEL}: {event.get('status', '')}{share}".ljust(72), end="", file=sys.stderr, flush=True)
        print(f"\r  AI    {MODEL} ready".ljust(72), file=sys.stderr)
    except (urllib.error.URLError, OSError, ValueError, LLMError) as exc:
        print(f"\n  AI    pulling {MODEL} failed: {exc}", file=sys.stderr)
    finally:
        _pulling.clear()
        _checked = (0.0, None)


def chat(system, user, json_mode=False):
    """The model's reply text. Raises LLMError when the call fails."""
    body = {"model": MODEL, "stream": False, "options": {"temperature": 0, "num_ctx": CONTEXT},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["format"] = "json"
    request = urllib.request.Request(f"{URL}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())["message"]["content"] or ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise LLMError(f"{MODEL} at {URL}: HTTP {exc.code}: {str(detail)[:200]}") from exc
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        raise LLMError(f"{MODEL} at {URL}: {exc}") from exc


def chat_json(system, user):
    """The reply parsed as JSON; the first {...} block when the model wraps it in text."""
    text = chat(system, user, json_mode=True)
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(match.group(0)) if match else {}
        except ValueError:
            return {}
