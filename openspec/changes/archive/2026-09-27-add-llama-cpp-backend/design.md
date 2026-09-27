# Design: add-llama-cpp-backend

## Context

Facts established before design (2026-09-27, desktop RTX 4060 8 GiB, WSL2), from the current code and from a `llama-server` build b11217 (commit c9064dded, the official `ubuntu-cuda-13.4-x64` release plus its CUDA runtime bundle) run by hand against `Qwen/Qwen2.5-0.5B-Instruct-GGUF` Q4_K_M:

- `api/deps.py::_build_engine_pool` constructs `VLLMEngine` directly, after vLLM-specific `apply_tier_knobs` and `validate_model_fits`. `ModelEntry.engine` is `Literal["vllm"]`.
- `BaseEngine` declares `generate`, `generate_stream`, `is_healthy`, `count_prompt_tokens` (optional, `None` means "cannot count") and `supports_tools`. Admission discovers `kv_capacity_tokens` with `getattr` and skips the KV gate with a `degraded` warning when it is absent.
- Manifest assembly hardcodes `backend_version=package_version("vllm")`, `ManifestEngine.backend` defaults to `"vllm"`, and `hardware.cuda` is `torch.version.cuda`.
- `llama-server` accepts and silently ignores unknown request fields, and ignores `model`.
- Its defaults apply `top_k=40` and `min_p=0.05`; vLLM applies neither. Other llama.cpp samplers (DRY, XTC, typical, mirostat) are off by default.
- Non-streaming and streaming responses carry `usage`. With `stream_options.include_usage` the stream ends with a `choices: []` usage chunk, then `[DONE]`. Responses also carry `timings` (`prompt_ms`, `predicted_ms`, `cache_n`), but nothing measures time spent waiting before the slot started.
- `POST /v1/chat/completions/input_tokens` returns the templated prompt's token count (35 for a request whose completion reported `prompt_tokens: 35`).
- A prompt longer than the context returns HTTP 400 `{"error": {"type": "exceed_context_size_error", "n_prompt_tokens": ..., "n_ctx": ...}}`. A `max_tokens` beyond the remaining context just stops at the end of the context.
- `GET /props` returns `build_info` (`b11217-c9064dded`), `model_path`, `model_ftype` (`Q4_K - Medium`), `chat_template`, `total_slots`, and per-slot `n_ctx`. `GET /health` is 503 while loading and 200 when ready.
- `llama-fit-params -m <gguf> -c <ctx>` prints the arguments that fit the model on the detected GPU in about half a second: `-c 8192 -ngl -1` (all layers) for Qwen2.5-Coder-7B Q4_K_M at 8192, `-c 131072 -ngl 14` at 131072.
- `llama-server` fits only arguments that were not set. InferenceX always passes `-c`, so the context it asked for is never shrunk behind its back.

## D1. External process, owned by the engine

`LlamaCppEngine` starts one `llama-server` in its constructor and owns it for the engine's lifetime:

- Binary: `INFERENCE_X_LLAMA_SERVER` if set, otherwise `llama-server` on `PATH`. Missing either way is a startup error naming both.
- Arguments: `-m <gguf> -c <max_model_len> -np 1 --host 127.0.0.1 --port <free port> --no-webui -ctk f16 -ctv f16 --no-cache-prompt`, plus `-ngl <n>` only when configured. Cache type and prompt caching are passed explicitly so the manifest can report them from what was launched instead of from documentation defaults.
- Prompt caching is off. `llama-server` enables it by default, and in the first live run the same request (temperature 0, same seed) returned different text depending on whether the prompt prefix came from the cache: with 40 of 41 prompt tokens reused, the greedy continuation diverged at the second word. A result that depends on which request ran before it breaks the comparability `run_id` promises, and InferenceX's vLLM path already runs with prefix caching off (the VRAM tier setting). The cost is that every turn re-processes its whole prompt; that is measured in the live validation.
- Port: a free port chosen by binding port 0 and closing it. There is a small window in which another process could take the port; the process would then fail to bind and exit, which surfaces as a startup failure, not a silent misroute.
- Readiness: poll `/health` until 200. If the process exits first, or a timeout (`INFERENCE_X_LLAMA_STARTUP_TIMEOUT_S`, default 300 s) passes, startup fails with the exit code and the last lines of the server log.
- Output: the server's stdout and stderr go to `logs/llama-server-<model>.log`.
- Shutdown: `terminate`, wait, then `kill`. The child is also started with a Linux parent-death signal (`prctl(PR_SET_PDEATHSIG, SIGTERM)`), so if InferenceX itself dies the kernel stops `llama-server` and its VRAM is released. There are no restarts; that is V1-2.

vLLM stays a required dependency (DEC-007, which keeps vLLM required, is unchanged). No llama.cpp Python binding is added.

## D2. Factory, and planning per backend

`engines/registry.py::create_engine(entry, tier, free_vram_gib, total_vram_gib)` is the one construction path. It is a two-branch dispatch on `entry.engine`, not a plugin registry:

- `vllm`: exactly today's code, moved from `api/deps.py`: `apply_tier_knobs`, `validate_model_fits`, `VLLMEngine(...)`.
- `llama_cpp`: `LlamaCppEngine(entry)`. vLLM's tier knobs and VRAM estimator do not apply; llama.cpp's own fitting does.

`/v1/plan` and `/v1/doctor` gain a `backend` field per entry. vLLM entries are computed as before. llama.cpp entries:

- Plan: `estimated_weight_gib` is the GGUF file's size when the file is in the local cache, else null (plan never downloads). The vLLM estimates (`estimated_kv_cache_gib`, `estimated_footprint_gib`, `gpu_memory_utilization`, block size, KV dtype, batched tokens) are null because they are not llama.cpp quantities. `gpu_layers` is what `llama-fit-params` reports for the configured context (-1 meaning all layers), or null when the binary or file is missing.
- Doctor: `fits` is true only when `llama-fit-params` places every layer on the GPU at the configured context. Partial offload is reported as not fitting, with the layer count in `reason`, because it runs but far slower than the plan assumes. A missing binary or GGUF file is `fits: false` with that reason.

`/v1/models` reports the GGUF file size as `estimated_weights_gib` for llama.cpp entries when the file is cached, and null otherwise.

## D3. Configuration

`ModelEntry.engine: Literal["vllm", "llama_cpp"]`. For `llama_cpp`:

- `model_path` is a Hugging Face repo id with `gguf_file` naming the file in it, or a local path ending in `.gguf` with `gguf_file` unset. The repo form is resolved through the Hugging Face cache (`hf_hub_download`), so a first start downloads the file, the same way vLLM entries download weights.
- `max_model_len` is required; it is the context `llama-server` is started with.
- `max_num_seqs` must be 1 (it defaults to 1 for these entries). One slot keeps admission's concurrency gate equal to what the server can actually run.
- `n_gpu_layers` is optional.
- Setting a vLLM-only field (`gpu_memory_utilization`, `max_num_batched_tokens`, `quantization`, `tool_call_parser`) on a llama.cpp entry is a validation error. `quantization` is included because the GGUF file determines it; the manifest reports the server's own value (D6).

`gguf_file` and `n_gpu_layers` on a vLLM entry are also validation errors.

## D4. Requests and sampling

The engine sends `messages` (role and content), `max_tokens` (always set: admission resolves it), `temperature`, `top_p`, `seed`, `stop`, `stream`, and `repeat_penalty` when the entry sets `repetition_penalty`. It also sends `top_k: 0` and `min_p: 0` (both mean disabled in llama.cpp) so the effective sampler set matches what InferenceX reports in `resolved`. Streaming always asks for usage internally; whether the client sees it is still decided by the client's `stream_options`.

The chat template is the one embedded in the GGUF, rendered by `llama-server` (Jinja is its default). `chat_template_sha256` is the sha256 of the template text `/props` returns.

Tool requests never reach the engine: `supports_tools` is false and `ChatService` returns 400 `tool_calling_unsupported` before admission (DEC-065, the tool-calling decision, already does this for any engine without a parser).

## D5. Capabilities

- `count_prompt_tokens` calls `/v1/chat/completions/input_tokens`. The contract is synchronous, so this is a blocking HTTP call to localhost of a few milliseconds from inside admission. With one model and one slot per process that pause is well below a decode step; if it ever matters, the contract can grow an async variant. On error it returns `None`, and admission reports `prompt_tokens_estimated` as it already does.
- `kv_capacity_tokens` is `n_ctx` from `/props`: with one slot, the KV cache holds exactly that many tokens.
- `is_healthy` is true while the process is running and was healthy at startup.
- A dead server is detected in `count_prompt_tokens`, which admission calls before a streaming response starts. That makes a streaming request to a dead server a 503 instead of a 200 whose stream breaks after the first event.

## D6. Provenance

The manifest reads backend identity from the engine instead of assuming vLLM. Each engine exposes a `manifest_identity` mapping. `VLLMEngine` does not gain one, and absence means "vLLM, as before", so vLLM manifests and `run_id`s are byte-for-byte unchanged. For llama.cpp:

- `engine.backend = "llama.cpp"`, `engine.backend_version` = `build_info` from `/props` (the running server, not a package lookup).
- `model.hf_repo` and `model.hf_revision` (the snapshot commit the file was resolved from) for the repo form; both null for a local path.
- `model.gguf_file`: new, the file name. It enters the `run_id` preimage only when non-null, so vLLM preimages keep their shape (the same rule DEC-065 used for `tools_sha256`).
- `model.weights_sha256`: sha256 of the GGUF bytes, computed once at startup from the file that was passed to `-m`. For the repo form this equals the Hub's published LFS sha256, which is a cross-check, not a substitute.
- `model.quantization`: `model_ftype` from `/props`.
- `runtime.max_model_len` and `runtime.kv_capacity_tokens`: `n_ctx`. `runtime.kv_cache_dtype = "f16"` and `runtime.prefix_caching = false`, from the launch arguments. `runtime.batch_invariant = false`. The vLLM-only runtime fields (attention backend, CUDA graphs, eager mode, block size, prefix-cache hash) are null.
- `hardware.cuda` is null for llama.cpp runs. It is `torch.version.cuda`, which describes vLLM's runtime and not the CUDA build `llama-server` was compiled against, so reporting it would be wrong. `hardware` is not part of `run_id`.
- `timing` is null. `llama-server` reports prompt and generation milliseconds but no queue time, and `EngineTiming` requires all three. Reporting two real numbers and an invented third is exactly what the timing contract forbids.

## D7. Errors

| Condition | Response |
|---|---|
| Prompt longer than the context | Admission rejects first with 400 `context_length_exceeded`. If `llama-server` still reports `exceed_context_size_error` (for example when token counting fell back to the estimate), the engine raises the same error. |
| `llama-server` exited after startup | `EngineUnavailableError`, HTTP 503 `engine_unavailable`; `/health` returns 503. |
| Other `llama-server` HTTP errors | `RuntimeError` with the server's message, HTTP 500, as for vLLM failures. |
| Binary missing, GGUF missing, process exits during startup, health timeout | Startup error; InferenceX does not start serving. |
| `INFERENCE_X_DETERMINISTIC=1` with a llama.cpp model | Startup error: deterministic mode is implemented with vLLM batch invariance only. |

`EngineUnavailableError` lives in `engines/base.py` because a dead backend is not a llama.cpp concept.

## D8. What this does not add

No generic backend interface beyond the existing `BaseEngine`, no `execution/` package, no capability registry. The only contract additions are optional attributes discovered the way `kv_capacity_tokens` already is (`manifest_identity`), and one error type. The Backend Abstraction Principle in DEC-047 (no backend-neutral abstraction until two backends justify it) is honored: the factory and the provenance hook exist because two backends now need them, and nothing is generalized past what the two need.

## Follow-ups (not in this change)

- The InferenceX process runs the import-time WSL pinned-memory probe written for vLLM whatever the backend. For a GGUF model it opens a CUDA context of about 94 MiB (measured) that nothing uses. It was left alone to keep the vLLM path unchanged.
- Tool calling for GGUF models. `llama-server` has its own tool-call parsing, which would need the same transport-never-repair review DEC-065 gave the Hermes parser.
- Prompt caching as an opt-in per model, for agent loops where re-processing the prompt every turn costs more than reproducibility is worth.
- More than one sequence per `llama-server`.

## Validation plan

- Unit: config validation, request translation, response and stream translation (usage, finish reasons, empty deltas), error mapping, provenance (llama.cpp manifest fields, vLLM `run_id` golden unchanged), factory dispatch, plan/doctor for both backends.
- Process lifecycle against a fake `llama-server` script: ready, early exit, startup timeout, death after startup, shutdown.
- Live on the RTX 4060: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` Q4_K_M as the smoke model and `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` Q4_K_M at 8192 context as the useful one. Non-streaming and streaming chat, usage, `/v1/models`, plan and doctor, an over-long prompt, a tool request, the manifest (sha256 against the Hub value), killing `llama-server` mid-run, and Aider's acceptance script against the 7B.

## Validation results (2026-09-27, RTX 4060 8 GiB, WSL2, llama.cpp b11217)

| Check | Qwen2.5-0.5B Q4_K_M | Qwen2.5-Coder-7B Q4_K_M |
|---|---|---|
| Startup (process start to healthy, includes sha256 of the file) | 4.1 s | 7.2 s (hashing 3.1 s) |
| `weights_sha256` equals the Hub's published LFS sha256 | yes | yes |
| VRAM in use at 8192 context | 1.1 GiB | 5.3 GiB |
| Non-streaming chat, `usage`, `resolved`, `run_id` | pass | pass |
| Streaming: prologue, content, terminal, usage event only with `include_usage` | pass | pass |
| Same request twice at temperature 0: same `run_id`, same text | pass (after turning prompt caching off) | pass |
| `/v1/models`, `/v1/plan` (`gpu_layers: -1`), `/v1/doctor` (`fits: true`) | pass | pass |
| Over-long prompt: 400 `context_length_exceeded` from admission | pass | pass |
| `tools`: 400 `tool_calling_unsupported`; `deterministic: true`: 400 | pass | pass |
| `llama-server` killed: chat and streaming 503 `engine_unavailable`, `/health` 503 | pass | |
| InferenceX stopped: `llama-server` stopped; InferenceX `kill -9`: `llama-server` gone, VRAM back to idle | pass | |
| Missing binary: startup fails with the reason, nothing left running | pass | |
| Aider 0.86.2, `scripts/aider_acceptance.py --model qwen2.5-coder-7b-gguf` | | 7/7 (4 streaming, 3 `--no-stream`), 7 requests all 200 |

Warm client-side performance of the 7B (a 1,168-token prompt, 256 output tokens, temperature 0, median of 5 after 2 warmups; measured from the client because the backend reports no engine timing): TTFT 451 ms, decode 54.9 tok/s, total 5.1 s. For reference, vLLM's AWQ 7B on the same card runs only at 4096 context and measured TTFT 525 ms, decode 60.2 tok/s.
