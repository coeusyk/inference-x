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
    # WSL2: FlashInfer sampler JIT needs a full CUDA toolkit; use PyTorch fallback.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    # FlashInfer JIT (used by vLLM sampling) needs nvcc; vllm bundles it under site-packages/nvidia/cu*/
    if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
      _nvcc="$(find .venv/lib -path '*/nvidia/cu*/bin/nvcc' -type f 2>/dev/null | sort | tail -1)"
      if [[ -n "$_nvcc" ]]; then
        export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
        export PATH="${CUDA_HOME}/bin:${PATH}"
      fi
    fi
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

Set INFERENCE_X_DEFAULT_MODEL to pick a single model from config/models.yaml.
Set INFERENCE_X_LOADED_MODELS (comma-separated) to load multiple models at once.
  Example: INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve
EOF
    ;;
esac
