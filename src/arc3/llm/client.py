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

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0) -> str:
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

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"] or ""

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

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0) -> str:
        t0 = time.time()
        out = self.inner.complete(messages, max_tokens=max_tokens,
                                  temperature=temperature, seed=seed)
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
    _table: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._table[rec["key"]] = rec["response"]

    @property
    def name(self) -> str:
        return f"replay:{self.path.name}"

    def complete(self, messages, *, max_tokens=512, temperature=0.0, seed=0) -> str:
        key = call_key(messages, max_tokens, temperature, seed)
        if key in self._table:
            self.hits += 1
            return self._table[key]
        self.misses += 1
        if self.fallback is None:
            raise ReplayMiss(key)
        return self.fallback.complete(messages, max_tokens=max_tokens,
                                      temperature=temperature, seed=seed)
