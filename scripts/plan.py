#!/usr/bin/env python3
"""Print the GET /v1/plan sizing report for every registered model.

Usage:
    make plan
    uv run python scripts/plan.py [--base-url http://localhost:8000]

Requires a running server (this script does not start one).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    url = f"{args.base_url}/v1/plan"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.load(resp)
    except urllib.error.URLError as exc:
        print(f"Could not reach {url}: {exc}", file=sys.stderr)
        print("Is the server running? (make chat / ./scripts/dev.sh serve)", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
