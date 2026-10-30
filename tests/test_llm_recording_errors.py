"""A failed or truncated call is written to the recording, and replay skips it."""

import json

import pytest

from arc3.llm import ReplayLLM, Truncated
from arc3.llm.client import RecordingLLM


class Boom:
    name = "boom"

    def __init__(self):
        self.n = 0
        self.last = {"usage": {"completion_tokens": 16000}}

    def complete(self, messages, **kw):
        self.n += 1
        if self.n == 1:
            raise Truncated("cut")
        return "ok"


def test_truncated_calls_are_recorded_and_not_replayed(tmp_path):
    p = tmp_path / "calls.jsonl"
    rec = RecordingLLM(Boom(), p)
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(Truncated):
        rec.complete(msgs, max_tokens=10)
    assert rec.complete(msgs, max_tokens=20) == "ok"
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["error"].startswith("Truncated") and lines[0]["usage"]["completion_tokens"] == 16000
    assert lines[1]["response"] == "ok"
    rp = ReplayLLM(p)
    assert rp.complete(msgs, max_tokens=20) == "ok"
