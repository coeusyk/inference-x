# Proposal: add-llama-cpp-backend

## Why

InferenceX serves every model through vLLM. That is the right engine for safetensors checkpoints on a CUDA GPU with room for its KV pool, but it is the wrong one for a large share of consumer setups: GGUF quantizations (Q4_K_M and friends) are what most local models ship as, vLLM's GGUF support is limited, and a 7B model at Q4 fits an 8 GiB card with a full 8k context under llama.cpp where the same model needs AWQ plus careful sizing under vLLM. `docs/INFERENCEX-EVOLUTION.md` lists "vLLM only" as a v1.0 blocker and names V1-1 as the fix: llama.cpp as a second backend, behind the same OpenAI-compatible endpoint.

The owner's statement of purpose for V1-1: add external `llama-server` as the second InferenceX execution backend so GGUF models can be served through the same OpenAI-compatible endpoint on hardware and model combinations where vLLM is not appropriate. The exit condition is that a useful GGUF model is served through InferenceX's existing API with truthful backend selection, planning, errors and provenance. `llama-server` starting is not the exit condition.

DEC-047 (the Engine Boundary decision) made backend plurality the long-term direction but deferred any second backend to a future accepted ADR, and `AGENTS.md` and `CONTRIBUTING.md` forbid one until then. This change is that ADR's implementation; the decision itself is DEC-067.

## What changes

- **A second engine, `LlamaCppEngine`.** It launches one `llama-server` process for the configured GGUF model, bound to `127.0.0.1` on a free port, waits for it to become healthy, and talks to it over HTTP. llama.cpp stays an external binary; no Python binding is added. The engine stops the process on shutdown, and the process is also killed by the kernel if InferenceX dies (Linux parent-death signal), so a crashed InferenceX cannot leave a `llama-server` holding VRAM.
- **An engine factory.** `engines/registry.create_engine` becomes the only place that constructs engines, dispatching on `ModelEntry.engine`. The vLLM path (tier knobs, VRAM fit check, `VLLMEngine`) moves into it unchanged. This also closes DEC-047's open exit criterion that `api/deps.py` must not construct `VLLMEngine` directly.
- **Config.** `ModelEntry.engine` accepts `llama_cpp`. A llama.cpp entry names a GGUF file: `model_path` is a Hugging Face repo id plus `gguf_file`, or a local `.gguf` path. `max_model_len` is required and becomes `llama-server`'s context size. V1-1 serves one sequence per process (`max_num_seqs: 1`). `n_gpu_layers` is optional; unset, `llama-server` fits the layers itself. vLLM-only settings on a llama.cpp entry, and `tool_call_parser`, are configuration errors instead of being ignored.
- **Same endpoint, same contract.** `/v1/chat/completions` streaming and non-streaming, `usage`, `resolved`, `warnings`, admission, `strict`, `/v1/models`, `run_id`. Requests are translated to exactly the fields InferenceX supports. llama.cpp samplers InferenceX does not expose (`top_k`, `min_p`) are sent as disabled so the sampling that runs is the sampling `resolved` reports.
- **Explicitly unsupported, never faked.** Tool calling (rejected with the existing 400 `tool_calling_unsupported`), deterministic mode (startup refuses; per-request refusal is unchanged), per-request engine timing (`timing` is null because `llama-server` reports no queue time), and native `/metrics` series.
- **Provenance.** The manifest records `backend: "llama.cpp"`, the running server's build id as `backend_version`, and the GGUF identity: repo, snapshot revision, file name, the sha256 of the file that was loaded, and the quantization the server reports. vLLM `run_id`s do not change.
- **Planning.** `/v1/plan` and `/v1/doctor` report a `backend` per model. llama.cpp entries are planned with llama.cpp's own `llama-fit-params` tool instead of vLLM's estimator, and a missing binary or GGUF file is reported, not guessed around.
- **Errors.** A context overflow is the existing 400 `context_length_exceeded`. A `llama-server` that has exited turns requests into 503 `engine_unavailable` and `/health` into 503. Startup failures (binary missing, file missing, the process exiting or never becoming healthy) stop InferenceX before it serves, with the tail of the server log in the error.

## Out of scope

- A model lifecycle supervisor, load/unload, or restarts (V1-2).
- More than one sequence per llama.cpp process, and more than one model per InferenceX process (one model per process stays, DEC-059).
- A third backend, a plugin mechanism, or backend-neutral execution types. The factory is a plain dispatch over two registered engines.
- Any vLLM change. The vLLM path is moved, not altered.
- Tool calling for GGUF models, proxying `llama-server`'s own Prometheus metrics, and GGUF downloads beyond the Hugging Face cache InferenceX already uses.
- Admission restoration, which stays deferred.
