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


def test_http_backend_sends_extra_fields_and_reads_content():
    """Qwen3.6 needs chat_template_kwargs to switch thinking off; it must reach the server."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    llm = OpenAIChatLLM(base_url=f"http://127.0.0.1:{srv.server_port}",
                        extra={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20})
    assert llm.complete(MSG, temperature=0.7) == "ok"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["top_k"] == 20 and seen["temperature"] == 0.7
    srv.server_close()


def _serve_once(reply: dict, seen: dict):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    return srv


def test_http_backend_sends_key_and_keeps_reasoning(tmp_path):
    import json
    seen: dict = {}
    srv = _serve_once({"choices": [{"finish_reason": "stop", "message": {
        "content": "{}", "reasoning_content": "because"}}]}, seen)
    llm = OpenAIChatLLM(base_url=f"http://127.0.0.1:{srv.server_port}", api_key="k-123")
    assert "k-123" not in repr(llm)
    rec = RecordingLLM(llm, tmp_path / "c.jsonl")
    assert rec.complete(MSG) == "{}"
    assert seen["auth"] == "Bearer k-123"
    row = json.loads((tmp_path / "c.jsonl").read_text())
    assert row["reasoning"] == "because" and row["finish_reason"] == "stop"
    assert "k-123" not in (tmp_path / "c.jsonl").read_text()
    srv.server_close()


def test_http_backend_raises_on_truncation():
    from arc3.llm import Truncated
    srv = _serve_once({"choices": [{"finish_reason": "length", "message": {"content": '{"a": "cut'}}]}, {})
    llm = OpenAIChatLLM(base_url=f"http://127.0.0.1:{srv.server_port}")
    with pytest.raises(Truncated):
        llm.complete(MSG)
    srv.server_close()


def test_http_backend_retries_server_errors():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            hits.append(1)
            if len(hits) == 1:
                self.send_response(500)
                self.end_headers()
                return
            out = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=lambda: [srv.handle_request() for _ in range(2)], daemon=True)
    t.start()
    llm = OpenAIChatLLM(base_url=f"http://127.0.0.1:{srv.server_port}", backoff=0.01)
    assert llm.complete(MSG) == "ok" and len(hits) == 2
    srv.server_close()
