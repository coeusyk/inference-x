#!/usr/bin/env python3
"""Logging pass-through proxy for capturing what an OpenAI client sends and receives.

Forwards every request to the upstream server unchanged, streams the response
back as it arrives (SSE included), and appends one JSON line per exchange:
the request path and body, the response status, and the raw response text.
Used to record acceptance-client traffic (Continue, Aider) as evidence.

    uv run python scripts/log_proxy.py --listen 8100 --upstream http://127.0.0.1:8000 --out capture.jsonl
    # then point the client at http://127.0.0.1:8100/v1
"""
from __future__ import annotations

import argparse
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

_lock = threading.Lock()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", type=int, default=8100)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True, help="JSONL file to append exchanges to")
    args = parser.parse_args()
    upstream = urlsplit(args.upstream)

    class Handler(BaseHTTPRequestHandler):
        def _forward(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            conn = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            received = bytearray()
            while chunk := resp.read1(65536):
                received += chunk
                self.wfile.write(chunk)
                self.wfile.flush()
            try:
                request_json = json.loads(body) if body else None
            except ValueError:
                request_json = body.decode("utf-8", "replace")
            record = {
                "method": self.command,
                "path": self.path,
                "request": request_json,
                "status": resp.status,
                "response": received.decode("utf-8", "replace"),
            }
            with _lock, open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        do_GET = do_POST = _forward

        def log_message(self, fmt: str, *a: object) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", args.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
