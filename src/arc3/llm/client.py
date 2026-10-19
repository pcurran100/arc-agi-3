"""LLM backends. See the package docstring for what each is for."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, Union
from urllib.parse import urlparse

#: A chat message: {"role": "system" | "user" | "assistant", "content": str}.
Message = dict

#: Hosts an HTTP backend may talk to without ``allow_remote``. Kaggle's
#: evaluation has no internet (Guardrail 5); a remote host here would work in
#: development and fail silently in the only run that counts.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class Truncated(RuntimeError):
    """The server stopped at ``max_tokens`` (``finish_reason == "length"``).

    A cut-off answer is never parsed: in the 2026-10-16 Kaggle run, 34 of 45 v3
    answers were unusable, mostly because thinking ran past the budget and the
    JSON was cut mid-string. Callers retry with more room or a terser prompt.
    """


class LLM(Protocol):
    """Chat completion. Deterministic given ``seed`` where the backend allows."""

    @property
    def name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
        schema: Optional[dict] = None,
    ) -> str: ...


def call_key(messages: Sequence[Message], max_tokens: int, temperature: float, seed: int) -> str:
    """Stable identity of one call, for recording and replay."""
    blob = json.dumps(
        {"m": list(messages), "t": max_tokens, "temp": temperature, "seed": seed},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


# -- test backend -------------------------------------------------------------


@dataclass
class ScriptedLLM:
    """Returns canned responses: a list consumed in order, or a function.

    A function receives the messages and returns the response, so a test can
    answer according to what it was asked.
    """

    responses: Union[Sequence[str], Callable[[Sequence[Message]], str]] = ()
    calls: list[list[Message]] = field(default_factory=list)
    _i: int = 0

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0, schema=None) -> str:
        self.calls.append(list(messages))
        if callable(self.responses):
            return self.responses(messages)
        if self._i >= len(self.responses):
            return ""
        r = self.responses[self._i]
        self._i += 1
        return r


# -- real backend ---------------------------------------------------------------


@dataclass
class OpenAIChatLLM:
    """An OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Written against llama.cpp's ``llama-server``; vLLM exposes the same route.
    """

    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    timeout: float = 600.0
    allow_remote: bool = False
    #: Extra fields merged into every request body, for server-specific options.
    #: Qwen3.6 thinks by default; ``{"chat_template_kwargs": {"enable_thinking":
    #: false}}`` turns that off in vLLM, and the model card's non-thinking
    #: sampling (top_p, top_k, presence_penalty) goes here too.
    extra: dict = field(default_factory=dict)
    #: Bearer token for hosted APIs (NVIDIA NIM, OpenRouter, ...) in development.
    #: Kept out of repr so it cannot leak into logs or call recordings.
    api_key: str = field(default="", repr=False)
    #: What the last call returned besides the answer: finish_reason, the
    #: reasoning trace (vLLM and NIM return it as ``reasoning_content``), usage.
    last: dict = field(default_factory=dict, repr=False)
    #: Retries on HTTP 429/5xx and timeouts, with backoff. NVIDIA NIM's free
    #: tier answered 28 of 77 calls with HTTP 500 on 2026-10-19.
    retries: int = 3
    backoff: float = 5.0

    def __post_init__(self) -> None:
        host = urlparse(self.base_url).hostname or ""
        if not self.allow_remote and host not in LOCAL_HOSTS:
            raise ValueError(
                f"{self.base_url} is not local; evaluation has no internet "
                "(Guardrail 5). Pass allow_remote=True only for prototyping."
            )

    @property
    def name(self) -> str:
        return f"openai-chat:{self.model}@{self.base_url}"

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0,
                 schema: Optional[dict] = None) -> str:
        req_body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        req_body.update(self.extra)
        if schema is not None:
            # Constrained decoding: the server only emits JSON matching the
            # schema (vLLM via xgrammar; llama.cpp via its grammar support).
            req_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "output"), "schema": schema},
            }
        body = json.dumps(req_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
        out = self._send(req)
        choice = out["choices"][0]
        msg = choice["message"]
        self.last = {
            "finish_reason": choice.get("finish_reason"),
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "usage": out.get("usage"),
        }
        if choice.get("finish_reason") == "length":
            raise Truncated(f"answer cut off at max_tokens={max_tokens}")
        return msg["content"] or ""

    def _send(self, req) -> dict:
        import socket
        import urllib.error
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
            except (TimeoutError, socket.timeout, urllib.error.URLError):
                if attempt == self.retries:
                    raise
            time.sleep(self.backoff * (2 ** attempt))
        raise RuntimeError("unreachable")

    def ready(self) -> bool:
        """Is the server up? llama-server answers ``/health`` once loaded."""
        try:
            with urllib.request.urlopen(self.base_url.rstrip("/") + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


# -- recording and replay ---------------------------------------------------------


@dataclass
class RecordingLLM:
    """Wraps a backend and appends every call to a JSONL file.

    Each line: key, backend name, the full request, the response, wall seconds.
    Written as it goes, so a run killed by a Kaggle timeout keeps what it did.
    """

    inner: LLM
    path: Path
    n_calls: int = 0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def last(self) -> dict:
        return getattr(self.inner, "last", None) or {}

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0, schema=None) -> str:
        t0 = time.time()
        out = self.inner.complete(messages, max_tokens=max_tokens,
                                  temperature=temperature, seed=seed, schema=schema)
        dt = time.time() - t0
        self.n_calls += 1
        self.seconds += dt
        rec = {
            "key": call_key(messages, max_tokens, temperature, seed),
            "backend": self.inner.name,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "response": out,
            "seconds": round(dt, 3),
        }
        last = getattr(self.inner, "last", None)
        if last:
            rec.update(last)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out


class ReplayMiss(KeyError):
    """A call that is not in the recording, with no fallback backend."""


@dataclass
class ReplayLLM:
    """Serves recorded responses by call key; optionally falls back to a backend.

    Replay is exact only while the prompts are byte-identical, so any change to
    scene rendering or prompt text turns hits into misses. ``hits``/``misses``
    make that visible rather than silent.
    """

    path: Path
    fallback: Optional[LLM] = None
    hits: int = 0
    misses: int = 0
    #: finish_reason and reasoning of the last call served, as a live backend has.
    last: dict = field(default_factory=dict)
    _table: dict[str, str] = field(default_factory=dict)
    _extra: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._table[rec["key"]] = rec["response"]
                    self._extra[rec["key"]] = {k: rec[k] for k in ("finish_reason", "reasoning")
                                               if k in rec}

    @property
    def name(self) -> str:
        return f"replay:{self.path.name}"

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0, schema=None) -> str:
        key = call_key(messages, max_tokens, temperature, seed)
        if key in self._table:
            self.hits += 1
            self.last = self._extra.get(key, {})
            return self._table[key]
        self.misses += 1
        if self.fallback is None:
            raise ReplayMiss(key)
        out = self.fallback.complete(messages, max_tokens=max_tokens,
                                     temperature=temperature, seed=seed, schema=schema)
        self.last = getattr(self.fallback, "last", None) or {}
        return out
