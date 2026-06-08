# InferenceX — common commands (run `make help` for full list)
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "InferenceX — available commands"
	@echo ""
	@echo "Server & playground"
	@echo "  make playground              Start server + Textual TUI (Chat tab)"
	@echo "  make playground-compare      Compare mode: MODEL_A=... MODEL_B=..."
	@echo "  make client                  Batch CLI (playground/client.py)"
	@echo "  make stop                    Stop background uvicorn process"
	@echo ""
	@echo "Benchmarks (server must be running)"
	@echo "  make benchmark MODEL=<name>  Run standard prompt suite for one model"
	@echo "  make benchmark-all           Benchmark qwen2.5-0.5b + tinyllama-chat"
	@echo "  make advise                  Print ranked model advisor report"
	@echo ""
	@echo "Development"
	@echo "  ./scripts/dev.sh sync        Install/sync dependencies (uv sync)"
	@echo "  ./scripts/dev.sh serve       Start API server only"
	@echo "  ./scripts/dev.sh test        Run unit tests"
	@echo "  ./scripts/dev.sh smoke       HTTP smoke test against running server"
	@echo ""
	@echo "Examples"
	@echo "  INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve"
	@echo "  make playground"
	@echo "  make benchmark MODEL=qwen2.5-0.5b"
	@echo "  uv run python playground/app.py --allow-internal"
	@echo "  uv run python playground/client.py --allow-internal \"Hello\""
	@echo ""
	@echo "API endpoints: POST /v1/chat/completions  GET /health  GET /v1/models"
	@echo "               GET /v1/benchmark/results  GET /v1/benchmark/advise"

# Start server + Textual playground in one command
.PHONY: playground
playground:
	@echo "Starting InferenceX server in background..."
	uv run uvicorn inference_x.api.main:app --host 127.0.0.1 --port 8000 &
	@echo "Waiting for server to be ready..."
	@until curl -sf http://localhost:8000/health > /dev/null; do sleep 1; done
	@echo "Server ready. Launching playground..."
	uv run python playground/app.py --allow-internal

# Compare mode
.PHONY: playground-compare
playground-compare:
	uv run uvicorn inference_x.api.main:app --host 127.0.0.1 --port 8000 &
	@until curl -sf http://localhost:8000/health > /dev/null; do sleep 1; done
	uv run python playground/app.py --allow-internal --compare $(MODEL_A) $(MODEL_B)

# Batch CLI runner
.PHONY: client
client:
	uv run python playground/client.py --allow-internal

# Stop background server
.PHONY: stop
stop:
	pkill -f "uvicorn inference_x.api.main:app" || true

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
