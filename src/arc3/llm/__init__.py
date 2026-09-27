"""The LLM seam (D-035): one narrow interface, several interchangeable backends.

The LLM's only job in this system is to *propose* hypotheses as code. It never
chooses an action and never sees the engine. Everything it says is tested by
the rest of the machinery before it can influence anything.

Backends, all behind :class:`LLM`:

- :class:`ScriptedLLM` — canned responses, for tests. No model, no network.
- :class:`OpenAIChatLLM` — any OpenAI-compatible chat endpoint. In development
  that is llama.cpp's ``llama-server`` on localhost; on Kaggle it is the same
  server (or vLLM) running offline on the notebook's GPU. Standard library only.
- :class:`RecordingLLM` / :class:`ReplayLLM` — write every call to JSONL, and
  serve a recorded run back offline. One GPU run's thinking can then be reused
  to tune everything downstream at zero GPU cost.

**Guardrail 5.** Nothing here may reach the internet at evaluation. The HTTP
client refuses any host that is not local unless explicitly allowed.
"""

from .client import (
    LLM,
    Message,
    OpenAIChatLLM,
    RecordingLLM,
    ReplayLLM,
    ReplayMiss,
    ScriptedLLM,
    call_key,
)

__all__ = [
    "LLM",
    "Message",
    "OpenAIChatLLM",
    "RecordingLLM",
    "ReplayLLM",
    "ReplayMiss",
    "ScriptedLLM",
    "call_key",
]
