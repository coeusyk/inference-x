#!/usr/bin/env python3
"""V1-0 exit test: a real Aider session makes a correct code edit through InferenceX.

Builds a throwaway git repo holding a buggy `median()` plus a test that fails on
it, asks Aider (pointed at the running server) to fix the bug, then runs the test.
Passing means the edit is correct, not just that the HTTP traffic succeeded.

    # Terminal 1
    INFERENCE_X_DEFAULT_MODEL=qwen2.5-coder-1.5b ./scripts/dev.sh serve
    # Terminal 2 (aider installed separately, e.g. `uv tool install aider-chat`)
    uv run python scripts/aider_acceptance.py              # streaming (Aider default)
    uv run python scripts/aider_acceptance.py --no-stream  # non-streaming

Exit code 0 means the test fails before Aider's edit and passes after it.
Check the server log for any 4xx/5xx afterwards: unknown request fields are
rejected with 400, so a clean log also shows Aider sent nothing unsupported.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
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
    "median() in stats.py returns the wrong value for even-length lists. Fix it so it "
    "returns the average of the two middle values for even-length lists. Only edit stats.py."
)


def _run_test(repo: Path) -> bool:
    return subprocess.run([sys.executable, "test_stats.py"], cwd=repo).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen2.5-coder-1.5b")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ix-aider-") as tmp:
        repo = Path(tmp)
        (repo / "stats.py").write_text(BUGGY)
        (repo / "test_stats.py").write_text(TEST)
        for cmd in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "-c", "user.name=ix", "-c", "user.email=ix@localhost",
                     "commit", "-qm", "buggy median"]):
            subprocess.run(cmd, cwd=repo, check=True)

        if _run_test(repo):
            print("FAIL: test passes before the edit; the task proves nothing")
            return 1

        aider = [
            "aider", f"--model=openai/{args.model}", f"--openai-api-base={args.base_url}",
            "--openai-api-key=unused", "--yes-always", "--no-auto-commits",
            "--no-check-update", "--analytics-disable", "--no-show-model-warnings",
            "--no-gitignore", "--message", PROMPT, "stats.py",
        ]
        if args.no_stream:
            aider.append("--no-stream")
        rc = subprocess.run(aider, cwd=repo).returncode
        print(f"--- aider exit code {rc}; stats.py after the edit:")
        print((repo / "stats.py").read_text())

        if rc != 0 or not _run_test(repo):
            print("FAIL: Aider's edit did not make the test pass")
            return 1
    print("PASS: Aider made a correct edit through InferenceX")
    return 0


if __name__ == "__main__":
    sys.exit(main())
