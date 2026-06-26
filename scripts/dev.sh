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
    # Preserve shell/cmdline INFERENCE_X_* before sourcing .env (match Python dotenv override=False).
    _cli_default_model="${INFERENCE_X_DEFAULT_MODEL:-}"
    _cli_loaded_models="${INFERENCE_X_LOADED_MODELS:-}"
    if [[ -f "$ROOT/.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "$ROOT/.env"
      set +a
    fi
    [[ -n "$_cli_default_model" ]] && export INFERENCE_X_DEFAULT_MODEL="$_cli_default_model"
    [[ -n "$_cli_loaded_models" ]] && export INFERENCE_X_LOADED_MODELS="$_cli_loaded_models"
    export INFERENCE_X_DEFAULT_MODEL="${INFERENCE_X_DEFAULT_MODEL:-qwen2.5-0.5b}"
    # WSL2: FlashInfer sampler JIT needs a full CUDA toolkit; use PyTorch fallback.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    # FlashInfer JIT (used by vLLM sampling) needs nvcc; vllm bundles it under site-packages/nvidia/cu*/
    if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
      _nvcc="$(find .venv/lib -path '*/nvidia/cu*/bin/nvcc' -type f 2>/dev/null | sort | tail -1)"
      if [[ -n "$_nvcc" ]]; then
        export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
        export PATH="${CUDA_HOME}/bin:${PATH}"
      fi
    fi
    # Set to 0.0.0.0 only behind a reverse proxy with TLS
    _host="${INFERENCE_X_HOST:-127.0.0.1}"
    uv run uvicorn inference_x.api.main:app --host "$_host" --port 8000
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
  help    Show this message

Make targets (run from repo root):
  make help              Full command reference
  make playground        Start server + Textual TUI
  make benchmark MODEL=<name>   Run benchmark suite (server must be running)
  make advise            Print ranked model advisor report
  make stop              Stop background server

Examples:
  ./scripts/dev.sh sync
  ./scripts/dev.sh serve          # terminal 1
  ./scripts/dev.sh smoke          # terminal 2
  make benchmark MODEL=qwen2.5-0.5b # terminal 2 (after serve)

Set INFERENCE_X_DEFAULT_MODEL to pick a single model from config/models.yaml.
Set INFERENCE_X_LOADED_MODELS (comma-separated) to load multiple models at once.
  Example: INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve
EOF
    ;;
esac
