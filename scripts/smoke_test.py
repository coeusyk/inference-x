#!/usr/bin/env python3
"""Smoke test for InferenceX.

Run against a live server:

    # Terminal 1 — start server (must use project .venv):
    INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
    # or: INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8000

    # Terminal 2 — smoke test:
    uv run python scripts/smoke_test.py

Exit code 0 means all checks passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

_VLLM_MISSING_HINT = """
Hint: the server process is not using the project .venv (vllm missing in that Python).
Stop the server and restart with:
  INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
or:
  INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8000
Do not use bare `uvicorn` or `python` from pyenv/shims.
"""

_CONNECTION_HINT = """
Hint: no server is listening. Start one first:
  INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
First /health request loads the model and may take 2–4 minutes.
"""


def _hint_for_body(body: dict) -> str:
    msg = body.get("error", {}).get("message", "")
    if "vllm is not installed" in msg:
        return _VLLM_MISSING_HINT
    return ""


def _post(url: str, payload: dict) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    except urllib.error.URLError:
        return 0, None


def _get(url: str) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    except urllib.error.URLError:
        return 0, None


def run(base_url: str) -> bool:
    ok = True

    print(f"[1/2] GET {base_url}/health")
    status, body = _get(f"{base_url}/health")
    if status == 0:
        print("      FAIL  connection refused (no server listening)")
        print(_CONNECTION_HINT)
        ok = False
    elif status == 200 and body and body.get("status") in ("healthy", "degraded"):
        print(f"      PASS  status={body['status']}")
    else:
        print(f"      FAIL  http={status} body={body}")
        if body:
            print(_hint_for_body(body))
        ok = False

    print(f"[2/2] POST {base_url}/v1/chat/completions")
    payload = {
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    status, body = _post(f"{base_url}/v1/chat/completions", payload)
    if status == 0:
        print("      FAIL  connection refused (no server listening)")
        print(_CONNECTION_HINT)
        ok = False
    elif status == 200 and body and body.get("object") == "chat.completion":
        content = body["choices"][0]["message"]["content"]
        tokens = body["usage"]["total_tokens"]
        print(f"      PASS  tokens={tokens} reply={content!r}")
    else:
        print(f"      FAIL  http={status} body={body}")
        if body:
            print(_hint_for_body(body))
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceX smoke test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running server (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"Smoke test → {args.base_url}\n")
    passed = run(args.base_url)
    print()
    if passed:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
