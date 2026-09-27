# Design: add-model-lifecycle-supervisor

## Context

- DEC-059 (B6) made one model per OS process the rule and moved multi-model serving to orchestration. The only orchestrator today is `playground/server_control.py`: one server per model on consecutive ports, stopped with `pkill -f "uvicorn inference_x"` and `pkill -f "VLLM::EngineCore"`.
- A worker process already has everything a request needs: registry, routing with aliases, admission, both backends, manifests. Its startup is all-or-nothing: uvicorn only accepts connections after the lifespan hook has built the engine, and exits if that fails.
- Measured start times on the RTX 4060 (vLLM 0.30.0 and llama.cpp b11217): vLLM coder-1.5b 35 s warm and 69 s on a first launch of a configuration; llama.cpp Qwen2.5-Coder-7B Q4_K_M 7 s; llama.cpp 0.5B 4 s. On an 8 GiB card only one realistic model fits at a time.
- The comparison target is Ollama (evolution doc section 8), which loads on first request, keeps models resident for a `keep_alive` period, and evicts to make room.

## D1. Supervisor and workers are separate processes

The supervisor is a small FastAPI app (`inference_x/supervisor/`) that never imports vLLM, torch or llama.cpp code paths and never touches the GPU. A worker is the existing app, started as `python -m uvicorn inference_x.api.main:app --host 127.0.0.1 --port <free port>` with `INFERENCE_X_DEFAULT_MODEL=<canonical name>` and `INFERENCE_X_LOADED_MODELS` removed from its environment. Everything else in the supervisor's environment is inherited, so settings such as `INFERENCE_X_DETERMINISTIC` or `INFERENCE_X_LLAMA_SERVER` apply to every worker.

Workers are started with a Linux parent-death signal, like `llama-server` in V1-1, so a killed supervisor does not leave workers holding VRAM. A worker's own output goes to `logs/worker-<model>.log`.

Why not swap engines inside one process: DEC-059 exists because in-process multi-engine serving cost CUDA graphs and forced heuristics, and vLLM does not reliably return all GPU memory to the process on shutdown. A process boundary is the only unload that is guaranteed to free VRAM, and it is also the crash-isolation boundary V1-2 has to provide.

## D2. Routing and transparency

The supervisor reads the same `config/models.yaml` and resolves `model` with `ModelRegistry.canonical_name` (names and aliases). Family names (variant selection) are not resolved by the supervisor, because choosing a variant depends on VRAM at load time; a family name is a 404 like any unregistered name. Requests are forwarded with the client's body unchanged, so the worker applies its own validation, alias handling and admission. Responses are streamed back byte for byte with the worker's status and the headers clients depend on (`Content-Type`, `X-Run-Id`, `Retry-After`).

A request whose body is not JSON or has no string `model` is rejected by the supervisor with the same 400 shape the worker uses.

## D3. Loading

A request for a model with no running worker triggers a load and waits for it. Loads are serialized by one supervisor lock because there is one GPU. A load:

1. Evicts workers until one slot is free (D4).
2. Starts the worker and polls its `/health` until 200. The worker is failed if the process exits first or `INFERENCE_X_WORKER_STARTUP_TIMEOUT_S` (default 600 s, enough for a first vLLM launch with a download) passes.
3. On failure, the worker is stopped and the request gets 503 `model_load_failed` with the exit code and the last lines of the worker log. The model's state records the failure until the next load attempt.

Requests arriving for a model that is already loading wait on the same load instead of starting another.

## D4. Residency and eviction

`INFERENCE_X_MAX_LOADED_MODELS` (default 1) caps running workers. When a load needs a slot, the least recently used worker is stopped once it has no requests in flight. The supervisor counts in-flight requests per worker from the start of forwarding until the response has been fully sent, streaming included. If no worker drains within `INFERENCE_X_SWITCH_WAIT_S` (default 60 s), the load is abandoned and the request gets 503 `model_busy` with `Retry-After`. A worker is never stopped mid-request.

Stopping a worker sends SIGTERM (uvicorn's graceful shutdown runs the app's shutdown hook, which stops vLLM or `llama-server`), waits up to 30 s, then SIGKILL.

There is no idle-unload timer. Ollama unloads after 5 minutes by default, but a vLLM cold start here is 35 to 70 s, and a daily driver that unloads during a coffee break would pay that on the next request. Unloading is explicit (`POST /v1/lifecycle/unload`) or happens through eviction. A timer can be added later if a real workflow needs VRAM back while idle.

The cap is a count, not a VRAM budget. Each worker still runs its own backend's fit check when it starts, and a load that does not fit fails with that check's reason. Packing several models by estimated VRAM is left for later: the vLLM estimate is not reliable enough across processes to decide evictions with.

## D5. Crash isolation

The supervisor checks a worker's process when it is used and when `GET /v1/lifecycle` or `/health` is called. A worker that has exited is marked `failed` with its exit code and log tail, and its slot is freed. Requests being forwarded to it when it died fail with 503 `engine_unavailable`. The next request for that model starts a fresh worker. Nothing restarts on its own, so a model that crashes on load cannot put the supervisor into a restart loop.

## D6. Endpoints

| Endpoint | Served by |
|---|---|
| `POST /v1/chat/completions` | Forwarded to the model's worker, loading it if needed |
| `GET /v1/models`, `GET /v1/plan`, `GET /v1/doctor` | The supervisor, with the same route code as a worker (registry and hardware probes only; nothing is loaded) |
| `GET /v1/lifecycle` | The supervisor: every registered model with `state` (`unloaded`, `loading`, `loaded`, `failed`), and for a worker its pid, port, started time, last use, requests in flight, and for a failure the exit code and reason |
| `POST /v1/lifecycle/load` `{"model": ...}` | Loads the model as a request would, and returns its lifecycle entry |
| `POST /v1/lifecycle/unload` `{"model": ...}` | Stops the worker after its in-flight requests finish (bounded by the switch wait); 409 `model_busy` if they don't |
| `GET /health` | 200 while the supervisor runs, with each loaded model's worker health |

`/metrics` and `/v1/metrics` stay per worker; the lifecycle entry gives the port.

## D7. Errors

| Condition | Response |
|---|---|
| `model` not registered (or a family name) | 404 `model_not_found` |
| Worker failed to start | 503 `model_load_failed`, with exit code and log tail |
| No resident worker drained in time | 503 `model_busy`, `Retry-After` |
| Worker died while serving the request | 503 `engine_unavailable` |
| Anything the worker returns | Passed through unchanged |

## Validation plan

- Unit tests against fake worker processes (a small HTTP server standing in for the app): routing and aliases, 404, load on demand, concurrent requests sharing one load, eviction order and draining, `model_busy`, load failure and timeout, a worker killed mid-request and while idle, unload, streaming passthrough with headers, and cleanup on supervisor shutdown.
- Live on the RTX 4060: switch between a llama.cpp model and a vLLM model through one endpoint; run two small models resident with `INFERENCE_X_MAX_LOADED_MODELS=2` and kill one worker while the other keeps serving; kill the supervisor and confirm no worker or `llama-server` is left; Aider and Continue acceptance through the supervisor.
