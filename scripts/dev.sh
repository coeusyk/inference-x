#!/usr/bin/env bash
# InferenceX local dev helpers (WSL2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmd="${1:-help}"

case "$cmd" in
  sync)
    uv sync
    ;;
  serve)
    export INFERENCE_X_DEFAULT_MODEL="${INFERENCE_X_DEFAULT_MODEL:-qwen2.5-0.5b}"
    uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8000
    ;;
  test)
    uv run python -m pytest tests/unit/ -v
    ;;
  smoke)
    uv run python scripts/smoke_test.py "${@:2}"
    ;;
  help|*)
    cat <<EOF
Usage: ./scripts/dev.sh <command>

Commands:
  sync    Install/sync all dependencies (includes vllm)
  serve   Start the API server (uses project .venv via uv run)
  test    Run unit tests
  smoke   Run HTTP smoke test against a running server

Examples:
  ./scripts/dev.sh sync
  ./scripts/dev.sh serve          # terminal 1
  ./scripts/dev.sh smoke          # terminal 2

Set INFERENCE_X_DEFAULT_MODEL to pick a model from config/models.yaml.
EOF
    ;;
esac
