#!/usr/bin/env python3
"""Continue exit test: Continue CLI makes a correct code edit through InferenceX tool calls.

Builds a throwaway directory holding a buggy `median()` plus a test that fails
on it, runs Continue CLI headless (pointed at the running server) to fix the
bug, then runs the test and checks the test file itself was not touched.
Passing means the edit is correct, not just that the HTTP traffic succeeded.

    # Terminal 1
    INFERENCE_X_DEFAULT_MODEL=qwen3-4b-fp8 ./scripts/dev.sh serve
    # Terminal 2 (Continue CLI: npm i -g @continuedev/cli)
    uv run python scripts/continue_acceptance.py --trials 3

To capture the traffic, run scripts/log_proxy.py in front of the server and
pass its URL with --api-base. Exit code 0 means every trial passed.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BUGGY = '''\
def median(xs):
    """Return the median of a non-empty list of numbers."""
    s = sorted(xs)
    return s[len(s) // 2]
'''

TEST = '''\
from stats import median

assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
assert median([5]) == 5
assert median([1, 2]) == 1.5
print("OK")
'''

PROMPT = (
    "test_stats.py fails because median() in stats.py is wrong for even-length "
    "lists. Fix median() in stats.py. Do not change test_stats.py."
)

CONFIG = """\
name: inferencex
version: 0.0.1
schema: v1
models:
  - name: {model}
    provider: openai
    model: {model}
    apiBase: {api_base}
    apiKey: unused
    roles: [chat, edit, apply]
    capabilities: [tool_use]
"""


def _test_passes(workdir: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "test_stats.py"], cwd=workdir, capture_output=True, text=True
    )
    return result.returncode == 0


def trial(model: str, api_base: str, timeout_s: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="ix-continue-") as tmp:
        workdir = Path(tmp)
        (workdir / "stats.py").write_text(BUGGY)
        (workdir / "test_stats.py").write_text(TEST)
        test_digest = hashlib.sha256(TEST.encode()).hexdigest()
        config = workdir / ".continue-config.yaml"
        config.write_text(CONFIG.format(model=model, api_base=api_base))

        if _test_passes(workdir):
            print("setup error: test passes before the edit")
            return False

        started = time.monotonic()
        try:
            run = subprocess.run(
                ["cn", "-p", PROMPT, "--config", str(config), "--auto"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            status = f"cn exit {run.returncode}"
        except subprocess.TimeoutExpired:
            status = f"cn timed out after {timeout_s}s"
        elapsed = time.monotonic() - started

        test_untouched = (
            hashlib.sha256((workdir / "test_stats.py").read_bytes()).hexdigest() == test_digest
        )
        passed = test_untouched and _test_passes(workdir)
        print(
            f"{'PASS' if passed else 'FAIL'}: {status}, {elapsed:.0f}s, "
            f"test file untouched: {test_untouched}"
        )
        if not passed:
            print("--- stats.py after the run ---")
            print((workdir / "stats.py").read_text())
        return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-4b-fp8")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600, help="seconds per trial")
    args = parser.parse_args()

    results = [trial(args.model, args.api_base, args.timeout) for _ in range(args.trials)]
    print(f"{sum(results)}/{len(results)} trials passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
