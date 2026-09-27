# Proposal: add-model-lifecycle-supervisor

## Why

The v1.0 thesis in `docs/INFERENCEX-EVOLUTION.md` is one OpenAI-compatible endpoint serving multiple registered models, with on-demand model lifecycle across vLLM and llama.cpp. Today an InferenceX process serves exactly the one model it was started with (DEC-059, "B6": one model per process, chosen because several engines in one process cost CUDA graphs, context length and hand-tuned VRAM heuristics). Changing models means stopping the server and starting another. The playground works around this by starting one server per model on consecutive ports and stopping them with `pkill -f`, which the architecture review listed as fragile coupling (process management by command-line string matching). Section 12 of the evolution doc lists "No runtime model switching" as a v1.0 blocker for V1-2, and "crash isolation not validated" as something V1-2 must address.

V1-2 adds the missing layer above one-model-per-process: a supervisor that owns one endpoint, starts a worker InferenceX process for a model when a request needs it, stops workers to make room, and keeps a crash in one worker from affecting anything else.

## What changes

- **A supervisor app**, `inference_x.supervisor.app:app`, started with `./scripts/dev.sh supervise`. It serves the OpenAI endpoint on port 8000 and holds no model and no CUDA context itself.
- **Workers are unchanged InferenceX processes.** A worker is `uvicorn inference_x.api.main:app` with `INFERENCE_X_DEFAULT_MODEL` set to one registered model, bound to `127.0.0.1` on a free port, owned by the supervisor (stopped with it, including when the supervisor dies). Everything a worker does today stays in the worker: backend selection, admission, `resolved`, `warnings`, manifests, `run_id`, timing. The supervisor forwards requests and responses without rewriting them.
- **Routing by `model`.** `POST /v1/chat/completions` goes to the worker for the request's `model`, resolved through the registry's names and aliases. An unregistered model is a 404 `model_not_found`, never a fallback to some other model.
- **On-demand loading.** A request for a model that is not loaded starts its worker and waits for it, bounded by the worker startup timeout. Concurrent requests for a model that is loading wait for the same load.
- **Residency.** At most `INFERENCE_X_MAX_LOADED_MODELS` workers run at once (default 1, the realistic number on an 8 GiB card). Loading one more first stops the least recently used worker that has no requests in flight. A worker is never stopped mid-request: if every resident worker is busy for longer than `INFERENCE_X_SWITCH_WAIT_S` (default 60 s), the new request gets 503 `model_busy` with `Retry-After`.
- **Explicit lifecycle API** for the V1-3 operator CLI: `GET /v1/lifecycle` (each registered model's state, and for running workers pid, port, uptime, requests in flight and last use), `POST /v1/lifecycle/load` and `POST /v1/lifecycle/unload`.
- **Crash isolation.** A worker that exits is detected and marked failed with its exit code and log tail. Its in-flight requests fail with 503 `engine_unavailable`, other workers keep serving, and the next request for that model starts a fresh worker. A worker that fails to start makes that request fail with 503 `model_load_failed` and the reason; there is no automatic restart loop.
- **Read-only endpoints** (`/v1/models`, `/v1/plan`, `/v1/doctor`) are served by the supervisor from the registry, without loading anything.

## Out of scope

- An idle-unload timer (Ollama's `keep_alive`). Models stay loaded until evicted or unloaded explicitly; see design D4.
- Serving several models from one process, or changing how a worker serves (DEC-059 stands).
- VRAM-aware packing of several resident models. Residency is a count; each worker still runs its own fit check when it starts.
- Moving the playground onto the supervisor, and the operator CLI itself (V1-3).
- Aggregating workers' `/metrics` and `/v1/metrics`. They remain available per worker.
- Admission restoration, which stays deferred.
