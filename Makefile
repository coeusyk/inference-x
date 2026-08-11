# InferenceX — common commands (run `make help` for full list)
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "InferenceX — available commands"
	@echo ""
	@echo "Server & playground"
	@echo "  make chat                    Start chat CLI (stops server on exit if it started one)"
	@echo "  make playground              Compare playground (stops server on exit if it started one)"
	@echo "  make playground-compare      Compare with preset models: MODEL_A=... MODEL_B=..."
	@echo "  make client                  Batch CLI (playground/client.py; does not start a server)"
	@echo "  make stop                    Stop background uvicorn / vLLM processes"
	@echo ""
	@echo "Benchmarks (server must be running)"
	@echo "  make benchmark MODEL=<name>  Run standard prompt suite for one model"
	@echo "  make benchmark-all           Benchmark qwen2.5-0.5b + tinyllama-chat"
	@echo "  make advise                  Print ranked model advisor report"
	@echo ""
	@echo "Diagnostics (server must be running)"
	@echo "  make plan                    Print per-model VRAM sizing plan (GET /v1/plan)"
	@echo "  make doctor                  Print environment + model-fit readiness (GET /v1/doctor)"
	@echo ""
	@echo "Development"
	@echo "  ./scripts/dev.sh sync        Install/sync dependencies (uv sync)"
	@echo "  ./scripts/dev.sh serve       Start API server only"
	@echo "  ./scripts/dev.sh test        Run unit tests"
	@echo "  ./scripts/dev.sh smoke       HTTP smoke test against running server"
	@echo ""
	@echo "Examples"
	@echo "  INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve"
	@echo "  make chat"
	@echo "  make playground"
	@echo "  make benchmark MODEL=qwen2.5-0.5b"
	@echo "  uv run python playground/app.py --allow-internal"
	@echo "  uv run python playground/client.py --allow-internal \"Hello\""
	@echo ""
	@echo "API endpoints: POST /v1/chat/completions  GET /health  GET /v1/models"
	@echo "               GET /v1/benchmark/results  GET /v1/benchmark/advise"
	@echo "               GET /v1/plan  GET /v1/doctor"

# Start server + Claude-style chat CLI (server starts after model selection in TUI)
.PHONY: chat
chat:
	@echo "Launching chat (server starts after you pick a model)…"
	uv run python playground/chat.py --allow-internal

# Compare two models side-by-side in the interactive playground
.PHONY: playground
playground:
	@echo "Launching compare playground (pick two models, server starts after selection)…"
	uv run python playground/app.py --allow-internal

# Compare with preset models (skips the two-model picker)
.PHONY: playground-compare
playground-compare:
	uv run python playground/app.py --allow-internal --compare $(MODEL_A) $(MODEL_B)

# Batch CLI runner
.PHONY: client
client:
	uv run python playground/client.py --allow-internal

# Stop background server
.PHONY: stop
stop:
	pkill -f '[u]vicorn inference_x.api.main:app' || true
	pkill -f 'VLLM::Engine[C]ore' || true

# Benchmark a single model (server must be running)
.PHONY: benchmark
benchmark:
	VLLM_OPT_LEVEL=O2 uv run python scripts/benchmark.py --model $(MODEL)

# Benchmark all small models (server must be running with INFERENCE_X_LOADED_MODELS set)
.PHONY: benchmark-all
benchmark-all:
	@for model in qwen2.5-0.5b tinyllama-chat; do \
		uv run python scripts/benchmark.py --model $$model; \
	done

# Print ranked advisor report from stored benchmark results
.PHONY: advise
advise:
	uv run python scripts/advise.py

# Print per-model VRAM sizing plan (GET /v1/plan)
.PHONY: plan
plan:
	uv run python scripts/plan.py

# Print environment + model-fit readiness (GET /v1/doctor)
.PHONY: doctor
doctor:
	uv run python scripts/doctor.py

# Compute the benchmark suite_version digest (add --write to update the file)
.PHONY: suite-version
suite-version:
	uv run python scripts/suite_version.py
