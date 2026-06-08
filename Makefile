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
	uv run python playground/client.py

# Stop background server
.PHONY: stop
stop:
	pkill -f "uvicorn inference_x.api.main:app" || true