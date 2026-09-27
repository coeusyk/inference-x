"""A stand-in for `llama-server`, for the llama.cpp engine's unit tests.

Speaks the subset of llama-server's HTTP API the engine uses. Behavior is set
with FAKE_LLAMA_MODE: "ok" (default), "exit" (exit 3 before listening),
"hang" (listen but never report healthy), "ctx" (chat requests fail with
llama-server's context-size error). Every chat request body is appended to
FAKE_LLAMA_RECORD as one JSON line, and the command line is written to
FAKE_LLAMA_RECORD + ".args".
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = os.environ.get("FAKE_LLAMA_MODE", "ok")
RECORD = os.environ.get("FAKE_LLAMA_RECORD")
USAGE = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            if MODE == "hang":
                self._json(503, {"error": {"code": 503, "message": "Loading model"}})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, {
                "default_generation_settings": {"n_ctx": 4096},
                "build_info": "b1-fake",
                "model_ftype": "Q4_K - Medium",
                "chat_template": "{{ messages }}",
                "total_slots": 1,
            })
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions/input_tokens":
            self._json(200, {"input_tokens": 7, "object": "response.input_tokens"})
            return
        if RECORD:
            with open(RECORD, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(body) + "\n")
        if MODE == "ctx":
            self._json(400, {"error": {
                "code": 400, "type": "exceed_context_size_error",
                "message": "request (5000 tokens) exceeds the available context size (4096 tokens)",
            }})
            return
        if not body.get("stream"):
            self._json(200, {
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "Hello there"}}],
                "usage": USAGE,
                "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0},
            })
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": USAGE},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> None:
    args = sys.argv[1:]
    if RECORD:
        with open(RECORD + ".args", "w", encoding="utf-8") as fh:
            json.dump(args, fh)
    if MODE == "exit":
        print("fake llama-server: failed to load model", flush=True)
        sys.exit(3)
    port = int(args[args.index("--port") + 1])
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
