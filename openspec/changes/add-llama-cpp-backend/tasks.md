# Tasks: add-llama-cpp-backend

## 1. Contract and config
- [x] 1.1 `EngineUnavailableError` in `engines/base.py`, mapped to 503 `engine_unavailable` (D7)
- [x] 1.2 `ModelEntry.engine` accepts `llama_cpp`; `gguf_file`, `n_gpu_layers`; per-backend field validation; one sequence for llama.cpp (D3)
- [x] 1.3 `ManifestModel.gguf_file`, in the run_id preimage only when set; recompute tests carry the rule; vLLM golden run_id unchanged (D6)

## 2. Engine and factory
- [x] 2.1 `engines/registry.create_engine`; vLLM construction moved from `api/deps.py` unchanged (D2)
- [x] 2.2 `LlamaCppEngine`: process start, health wait, log tail on failure, parent-death signal, shutdown (D1)
- [x] 2.3 GGUF resolution through the Hugging Face cache or a local path; sha256 of the loaded file (D6)
- [x] 2.4 Request translation with disabled llama-only samplers; response and stream translation; timing left null (D4, D6)
- [x] 2.5 Token counting via `/v1/chat/completions/input_tokens`; dead-server detection before streaming starts (D5)
- [x] 2.6 Context overflow maps to `context_length_exceeded`; dead server maps to `engine_unavailable` (D7)
- [x] 2.7 Deterministic startup refused for llama.cpp models (D7)
- [x] 2.8 Prompt caching off after live evidence that it changes greedy output (D1)

## 3. Provenance and planning
- [x] 3.1 Manifest reads `manifest_identity` and `runtime_snapshot`; `hardware.cuda` null for llama.cpp (D6)
- [x] 3.2 `utils/llama_cpp_plan.py`: binary lookup, cached GGUF path and size, `llama-fit-params` (D2)
- [x] 3.3 `/v1/plan`, `/v1/doctor`, `/v1/models` per backend (D2)

## 4. Tests
- [x] 4.1 Engine against a fake `llama-server` process: identity, arguments, translation, streaming, errors, startup failures, shutdown
- [x] 4.2 Config validation, factory dispatch, deterministic refusal, manifest, plan/doctor/models, 503 handler
- [x] 4.3 Unit suite, ruff, mypy, `openspec validate --strict`

## 5. Live validation and docs
- [x] 5.1 Qwen2.5-0.5B and Qwen2.5-Coder-7B Q4_K_M on the RTX 4060 (design, Validation results)
- [x] 5.2 Aider acceptance 7/7 on the 7B
- [x] 5.3 DEC-067; `AGENTS.md` and `CONTRIBUTING.md` anti-scope; README section and env vars; CHANGELOG
