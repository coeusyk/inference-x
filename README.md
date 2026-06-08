# InferenceX

InferenceX is a self-hosted LLM inference platform built incrementally on top of vLLM.
It provides an OpenAI-compatible `POST /v1/chat/completions` endpoint, a model registry,
an observability pipeline, and an interactive Textual playground — all designed to run on
a single WSL2 machine with one consumer-grade GPU.

## Current direction

The project begins with a stable OpenAI-compatible chat completions API and expands in phases:
- Phase 1: core vLLM-backed inference
- Phase 2: model registry and routing
- Phase 3: observability
- Phase 4: playground and evaluation
- Phase 5: hardening and publication readiness
- Phase 6: benchmark suite and model advisor (in progress)

## Key docs

- `docs/ARCHITECTURE.md`
- `docs/PHASES.md`
- `docs/DECISIONS.md`
- `.cursor/rules/`

---

## Prerequisites

### WSL2 + Ubuntu

InferenceX runs on WSL2 Ubuntu (tested on Ubuntu 22.04). Enable WSL2 in Windows and
install Ubuntu from the Microsoft Store. The GPU must be accessible inside WSL2:

```bash
nvidia-smi   # should return GPU info; if not, update your NVIDIA Windows driver
```

### CUDA Toolkit

vLLM requires CUDA. The recommended approach is to use the bundled CUDA wheels installed
by `uv sync`. If you need a system-level CUDA toolkit (e.g. for FlashInfer JIT), install
CUDA 12.x:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4
```

Then export `CUDA_HOME=/usr/local/cuda` in your shell profile.

> **WSL2 note:** FlashInfer JIT often fails on WSL2 due to missing system nvcc. InferenceX
> automatically sets `VLLM_USE_FLASHINFER_SAMPLER=0` on WSL2 to avoid this.

### uv

InferenceX uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/youruser/inference-x.git
cd inference-x

# Install all dependencies (creates .venv automatically)
uv sync

# Verify unit tests pass (no GPU required)
uv run pytest tests/unit -q
```

> **Important:** Always use `uv run` to invoke scripts and tools. Bare `python` or
> `uvicorn` from your shell PATH will bypass `.venv` and vllm will appear missing.

---

## Pre-downloading models

Models are downloaded from HuggingFace on first use. Pre-downloading avoids a silent
stall during server startup:

```bash
# Using huggingface-cli (recommended)
uv run huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
uv run huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
uv run huggingface-cli download facebook/opt-125m

# Or using the Python API
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
"
```

For gated models (e.g. Llama 3) you must authenticate first:

```bash
uv run huggingface-cli login
uv run huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Quick start (WSL2)

```bash
# Terminal 1 — start server with one model
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — smoke test
uv run python scripts/smoke_test.py

# Or call the API directly
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"Hello"}]}'
```

### Loading multiple models (for compare mode)

```bash
INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve
```

The `INFERENCE_X_LOADED_MODELS` environment variable accepts a comma-separated list of
model names from `config/models.yaml`. All listed models are loaded at startup and served
simultaneously. VRAM is split automatically between them.

---

## Make targets

```bash
make playground        # Start server + Textual TUI in one command
make playground-compare MODEL_A=qwen2.5-0.5b MODEL_B=tinyllama-chat
make client            # Batch CLI runner (rich terminal output)
make stop              # Stop background uvicorn process
make benchmark MODEL=qwen2.5-0.5b   # Run benchmark suite (Phase 6)
make benchmark-all     # Benchmark all models in models.yaml (Phase 6)
make advise            # Print ranked model advisor report (Phase 6)
```

---

## Configuration reference

| File | Purpose |
|------|---------|
| `config/models.yaml` | Model registry — name, model_path, gpu_memory_utilization, max_model_len |
| `config/routing.yaml` | Routing policy — default_model, fallback chain |
| `config/server.yaml` | Server defaults — host (127.0.0.1), port (8000) |
| `config/logging.yaml` | Logging config — rotating file handler + console |

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_X_DEFAULT_MODEL` | `qwen2.5-0.5b` | Model to load when LOADED_MODELS is unset |
| `INFERENCE_X_LOADED_MODELS` | (default model) | Comma-separated list of models to load at startup |
| `INFERENCE_X_CONFIG_DIR` | `config` | Path to config directory |
| `INFERENCE_X_METRICS_FILE` | (unset) | If set, enables NDJSON metrics export to this path |
| `INFERENCE_X_STREAM_TIMEOUT_S` | `120` | Per-token SSE timeout in seconds (0 = disabled) |

---

## Benchmark (Phase 6 — coming soon)

The benchmark suite measures throughput, time-to-first-token (TTFT), and latency
percentiles (p50/p95/p99) for each loaded model and produces a hardware-aware ranking.

```bash
# Requires server to be running
make benchmark MODEL=qwen2.5-0.5b
make benchmark-all
make advise
```

Results are stored in `docs/benchmarks/` as JSON files and can be read via
`GET /v1/benchmark/results` and `GET /v1/benchmark/advise`.

> **Coming soon:** benchmark scoring, hardware profiler, and playground Benchmark tab
> are implemented in Phase 6 (in progress).

---

## Troubleshooting

### First-load stall

**Symptom:** Server appears to hang after printing `Using FlashAttention version 2`.

**Cause:** HuggingFace is downloading the model weights in the background with no
progress indicator (~2–7 GB depending on model). This is normal.

**Fix:** Pre-download the model weights before starting the server (see above). You can
watch the HuggingFace cache directory to see download progress:

```bash
watch -n 2 "du -sh ~/.cache/huggingface/hub/"
```

If the server was interrupted mid-download, clear the incomplete cache entry:

```bash
rm -rf ~/.cache/huggingface/hub/models--<org>--<model>
```

### CUDA not found

**Symptom:** `RuntimeError: vLLM FlashInfer JIT requires nvcc` or
`CUDA_HOME not set and nvcc not found`.

**Fix 1 (recommended):** Always start the server via `uv run` or `./scripts/dev.sh serve`.
The bundled `nvidia-cuda-nvcc` wheel provides nvcc in `.venv`.

**Fix 2:** Set `CUDA_HOME` to your CUDA toolkit root:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

**Fix 3 (WSL2-specific):** InferenceX automatically disables FlashInfer on WSL2
(`VLLM_USE_FLASHINFER_SAMPLER=0`). If you see this error anyway, run:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
./scripts/dev.sh serve
```

### Port already in use

**Symptom:** `OSError: [Errno 98] Address already in use` on port 8000.

**Fix:** Find and kill the existing process:

```bash
make stop
# or manually:
lsof -ti:8000 | xargs kill -9
```

A previous `make playground` may have left a background uvicorn process running.
Always run `make stop` before restarting.

### Two servers on one GPU (KV cache error)

**Symptom:** Second server fails with
`Available KV cache memory: -0.04 GiB` or similar negative cache error.

**Fix:** Only one vLLM server per GPU. Kill all existing processes first:

```bash
pkill -f "uvicorn inference_x"
pkill -f "VLLM::EngineCore"
```

---

## Development environment

- Windows host with WSL2 Ubuntu as the primary runtime
- Cursor as the editor (with agent-assisted development via `.cursor/rules/`)
- Python 3.13+ managed by uv
- vLLM for inference (must be run with `uv run`)
