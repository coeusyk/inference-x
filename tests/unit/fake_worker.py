"""A stand-in for an InferenceX worker process, for the supervisor's unit tests.

Serves `/health` and `/v1/chat/completions` on the port given with `--port`,
for the model in INFERENCE_X_DEFAULT_MODEL. A model listed in
FAKE_WORKER_FAIL exits with code 3 instead of starting. Request fields the
real API would reject drive test behavior: `sleep` (seconds before
answering) and `die` (exit mid-request).
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ["INFERENCE_X_DEFAULT_MODEL"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, json.dumps({"status": "healthy", "model": MODEL}).encode(), "application/json")

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(float(body.get("sleep", 0)))
        if body.get("die"):
            os._exit(1)
        if body.get("stream"):
            events = b"".join(
                f"data: {json.dumps({'model': MODEL, 'i': i})}\n\n".encode() for i in range(3)
            ) + b"data: [DONE]\n\n"
            self._send(200, events, "text/event-stream")
            return
        payload = {"model": MODEL, "pid": os.getpid(), "asked": body.get("model")}
        self._send(200, json.dumps(payload).encode(), "application/json", {"X-Run-Id": "sha256:fake"})


def main() -> None:
    if MODEL in os.environ.get("FAKE_WORKER_FAIL", "").split(","):
        print(f"fake worker: cannot load {MODEL}", flush=True)
        sys.exit(3)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
