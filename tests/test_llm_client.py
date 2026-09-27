"""The LLM seam: scripted, recording, replay, and the no-internet guard."""

from __future__ import annotations

import pytest

from arc3.llm import OpenAIChatLLM, RecordingLLM, ReplayLLM, ReplayMiss, ScriptedLLM, call_key

MSG = [{"role": "user", "content": "hello"}]


def test_scripted_in_order_then_empty():
    llm = ScriptedLLM(["a", "b"])
    assert [llm.complete(MSG) for _ in range(3)] == ["a", "b", ""]
    assert len(llm.calls) == 3


def test_scripted_function_sees_messages():
    llm = ScriptedLLM(lambda m: m[-1]["content"].upper())
    assert llm.complete(MSG) == "HELLO"


def test_record_then_replay_offline(tmp_path):
    path = tmp_path / "calls.jsonl"
    rec = RecordingLLM(ScriptedLLM(["x"]), path)
    assert rec.complete(MSG, seed=3) == "x"
    rep = ReplayLLM(path)
    assert rep.complete(MSG, seed=3) == "x"
    assert rep.hits == 1
    with pytest.raises(ReplayMiss):
        rep.complete(MSG, seed=4)  # different seed, different call


def test_replay_falls_back_and_counts_misses(tmp_path):
    rep = ReplayLLM(tmp_path / "none.jsonl", fallback=ScriptedLLM(["live"]))
    assert rep.complete(MSG) == "live"
    assert rep.misses == 1


def test_call_key_depends_on_every_parameter():
    base = call_key(MSG, 512, 0.0, 0)
    assert base != call_key(MSG, 256, 0.0, 0)
    assert base != call_key(MSG, 512, 0.5, 0)
    assert base != call_key(MSG, 512, 0.0, 1)


def test_http_backend_refuses_remote_hosts():
    with pytest.raises(ValueError):
        OpenAIChatLLM(base_url="https://api.example.com")
    OpenAIChatLLM(base_url="http://127.0.0.1:8080")  # local is fine
