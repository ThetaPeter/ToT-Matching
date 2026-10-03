"""OpenAI-compatible chat HTTP helper (xAI, OpenRouter, etc.)."""

from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any

_KEEPALIVE: dict[str, http.client.HTTPSConnection] = {}


def _http_timeout_s(timeout_s: float | None) -> float:
    if timeout_s is not None:
        return float(timeout_s)
    return float(os.environ.get("TOT_HTTP_TIMEOUT", "300"))


def _https_conn(host: str, timeout_s: float, *, reset: bool = False) -> http.client.HTTPSConnection:
    if reset:
        old = _KEEPALIVE.pop(host, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
    conn = _KEEPALIVE.get(host)
    if conn is None:
        conn = http.client.HTTPSConnection(
            host,
            timeout=timeout_s,
            context=ssl.create_default_context(),
        )
        _KEEPALIVE[host] = conn
    return conn


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0.2,
    timeout_s: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> str:
    """POST ``/chat/completions`` and return the first message content string."""
    url = base_url.rstrip("/") + "/chat/completions"
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/chat/completions"
    timeout = _http_timeout_s(timeout_s)
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": float(temperature),
    }
    if extra_body:
        body.update(dict(extra_body))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Connection": "keep-alive",
        "Host": host,
    }
    if extra_headers:
        headers.update(dict(extra_headers))
    raw_body = json.dumps(body).encode("utf-8")
    payload: dict[str, Any] | None = None
    last_exc: Exception | None = None
    retryable_status = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(4):
        conn = _https_conn(host, timeout, reset=attempt > 0)
        try:
            conn.request("POST", path, body=raw_body, headers=headers)
            response = conn.getresponse()
            detail = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                last_exc = RuntimeError(f"HTTP {response.status} from {url}: {detail}")
                if response.status in retryable_status and attempt < 3:
                    time.sleep(2.0 * (2**attempt))
                    continue
                raise last_exc
            payload = json.loads(detail)
            break
        except (RemoteDisconnected, IncompleteRead, TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
            _https_conn(host, timeout, reset=True)
            if attempt < 3:
                time.sleep(2.0 * (2**attempt))
                continue
            raise RuntimeError(f"Failed to reach {url}: {exc}") from exc
    if payload is None:
        raise RuntimeError(f"Failed to reach {url}: {last_exc}")

    _append_usage_log(model=model, payload=payload)

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty chat completion response from {model}: {payload!r}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"Missing message content from {model}: {payload!r}")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _append_usage_log(*, model: str, payload: Mapping[str, Any]) -> None:
    path = os.environ.get("TOT_USAGE_LOG")
    if not path:
        return
    usage = payload.get("usage") or {}
    row = {
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        if isinstance(usage.get("completion_tokens_details"), Mapping)
        else usage.get("reasoning_tokens"),
    }
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def extract_json_object(text: str) -> Any:
    """Parse a JSON object from a model reply, tolerating fenced markdown."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    blob = stripped[start : end + 1] if start >= 0 and end > start else stripped
    try:
        return json.loads(_escape_invalid_json_backslashes(blob))
    except json.JSONDecodeError:
        return json.loads(blob)


def _escape_invalid_json_backslashes(text: str) -> str:
    """Double backslashes that are not legal JSON escapes (e.g. LaTeX \\frac)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\\" or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1]
        if nxt in '"\\/nrt':
            out.append(text[i : i + 2])
            i += 2
            continue
        if nxt == "u" and i + 6 <= n and all(
            c in "0123456789abcdefABCDEF" for c in text[i + 2 : i + 6]
        ):
            out.append(text[i : i + 6])
            i += 6
            continue
        out.append("\\\\")
        i += 1
    return "".join(out)
