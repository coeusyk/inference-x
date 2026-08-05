#!/usr/bin/env python3
"""Compute the benchmark suite `suite_version` digest (DEC-054).

Prints the pinned SHA-256 of a suite file's parsed prompt collection. With
`--write`, updates the suite file's `suite_version` field in place.

Usage:
    uv run python scripts/suite_version.py
    uv run python scripts/suite_version.py --suite benchmarks/prompts/standard.json
    uv run python scripts/suite_version.py --write
    make suite-version
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from inference_x.benchmarks.suite_identity import (  # noqa: E402
    DEFAULT_SUITE_PATH,
    compute_suite_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark suite_version.")
    parser.add_argument("--suite", default=DEFAULT_SUITE_PATH, help="Path to suite JSON")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the computed digest back into the suite file's suite_version",
    )
    args = parser.parse_args()

    path = Path(args.suite)
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = compute_suite_version(data["prompts"])

    if args.write:
        data["suite_version"] = digest
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote suite_version {digest} to {path}")
    else:
        print(digest)


if __name__ == "__main__":
    main()
