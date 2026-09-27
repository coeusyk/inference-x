# Tasks: add-model-lifecycle-supervisor

Paused 2026-09-27 on the RTX 4060 desktop; resume from the RTX 3060 6 GB laptop.

## 1. Implementation
- [x] 1.1 `utils/process.py` (free port, parent-death signal), shared with the llama.cpp engine
- [x] 1.2 `supervisor/workers.py`: worker processes, on-demand load, LRU eviction with drain, failure recording (D1 to D5)
- [x] 1.3 `supervisor/app.py`: forwarding proxy, `/v1/lifecycle` load/unload/list, read-only routes, `/health` (D6, D7)
- [x] 1.4 `./scripts/dev.sh supervise`
- [x] 1.5 Unit tests against fake worker processes (14 tests); unit suite, ruff, mypy, strict OpenSpec green

## 2. Live validation (RTX 4060, done so far)
- [x] 2.1 Supervisor holds no CUDA; llama.cpp and vLLM models switch through one endpoint with VRAM released on each eviction
- [x] 2.2 Two resident models: killing one worker leaves the other serving; its llama-server exits with it; next request reloads
- [x] 2.3 `kill -9` of the supervisor leaves no worker, EngineCore or llama-server
- [x] 2.4 Aider 7/7 through the supervisor (qwen2.5-coder-7b-gguf)
- [ ] 2.5 Continue through the supervisor: 7/10, 9/10, 9/10 against a same-day 10/10 run directly on a worker. Classified failures were model output (wrong unconditional average twice, a mismatched Edit old_string once); one run-1 failure had no capture. Decide whether more trials are needed

## 3. Remaining
- [ ] 3.1 README section and env vars (INFERENCE_X_MAX_LOADED_MODELS, INFERENCE_X_SWITCH_WAIT_S, INFERENCE_X_WORKER_STARTUP_TIMEOUT_S)
- [ ] 3.2 DEC entry, CHANGELOG, and evolution doc sequence
- [ ] 3.3 Re-run validation on the RTX 3060 6 GB laptop (the 6gb tier; smaller models)
- [ ] 3.4 PR into develop
