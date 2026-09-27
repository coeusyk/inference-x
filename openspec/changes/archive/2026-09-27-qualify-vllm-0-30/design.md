# Design: qualify-vllm-0-30

## Context

- **Old version:** vLLM 0.22.1 (what `uv.lock` resolved for `vllm>=0.6.0`; every live result recorded up to PR #42 used it).
- **Candidate:** vLLM 0.30.0, pinned exactly.
- **Host:** desktop, RTX 4060 8 GiB (SM 8.9), WSL2 kernel 6.18.33.2-microsoft-standard-WSL2, Python 3.13.2. The card resolves to the `6gb` VRAM tier because there is no tier between 0 and 10 GiB.
- **Date:** 2026-09-27.

Upstream changes between the two versions that could matter to InferenceX: PyTorch 2.13 (0.27), Transformers 5.15 (0.28), Model Runner V2 as the default model runner (0.29), a KV-cache watermark (0.24), the batch-invariant module moving under `model_executor/determinism/`, and a new supported WSL2 pinned-memory opt-in.

## D1. Coupling inventory

Every place InferenceX touches vLLM was checked against the 0.30.0 source before anything ran, then exercised live.

| Coupling | Where | 0.30.0 result |
|---|---|---|
| Engine construction: `AsyncEngineArgs` fields, `AsyncLLM.from_engine_args` | `engines/vllm_engine.py` | Unchanged |
| Generation: `AsyncLLM.generate`, `get_tokenizer`, `errored`, `shutdown` | `engines/vllm_engine.py` | Unchanged |
| `SamplingParams` fields used (temperature, max_tokens, top_p, repetition_penalty, seed, stop, skip_special_tokens) | `engines/vllm_engine.py` | Unchanged |
| Output objects: `RequestOutput`, `CompletionOutput` | `engines/vllm_engine.py` | Unchanged |
| Per-request timing: `RequestStateStats.queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts` | `engines/vllm_engine.py` | Unchanged; live `prefill + decode == inference` on every run |
| KV and config introspection: `AsyncLLM.vllm_config`, `CacheConfig` (`num_gpu_blocks`, `block_size`, `cache_dtype`, `enable_prefix_caching`, `prefix_caching_hash_algo`), `ModelConfig`, `AttentionConfig.backend` | `engines/vllm_engine.py` | Unchanged; manifest `kv_capacity_tokens` matches vLLM's own "GPU KV cache size" log line for all three models |
| Hermes tool parser: `ToolParserManager.get_tool_parser("hermes")`, `Hermes2ProToolParser.tool_call_start_token`, `extract_tool_calls` | `engines/vllm_engine.py` | Unchanged; Continue 10/10 |
| Env names: `VLLM_BATCH_INVARIANT`, `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_ENABLE_V1_MULTIPROCESSING`, `VLLM_USE_FLASHINFER_SAMPLER` | `utils/cuda_env.py`, `utils/determinism.py` | Unchanged |
| Batch invariance: `init_batch_invariance` | `utils/determinism.py` | **Moved** to `vllm.model_executor.determinism.batch_invariant` (D2) |
| WSL2 pinned memory | `utils/vllm_platform_patch.py`, `scripts/install_vllm_patch.sh` | **Old patch has no effect** (D3) |
| Native Prometheus metrics named in InferenceX code and docs (`request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `inter_token_latency_seconds`, `kv_cache_usage_perc`, `num_preemptions`, `num_requests_waiting_by_reason`, `prefix_cache_queries`) | `api/main.py` mount, docs | All still defined; `/metrics` exposes 100 `vllm:` families on 0.30.0 against 96 on 0.22.1 |

`VLLM_DETERMINISM_WARMUP_ITERATIONS`, which the original determinism plan named, still does not exist in 0.30.0, so the application-level warmup stays.

## D2. Batch-invariant import

`utils/determinism.py` imports `init_batch_invariance` from `vllm.model_executor.determinism.batch_invariant`. In 0.30.0 the function also resolves tuned matmul configs before enabling the mode. InferenceX does not depend on that detail; it still sets `VLLM_BATCH_INVARIANT=1` before engine construction and lets a real init failure propagate. The unit test that stubs the module now stubs the new path, because the old assertion was tied to an upstream module location that no longer exists. The GPU determinism conformance test (one unique sample over N trials) passed on 0.30.0.

## D3. WSL2 pinned memory

Evidence gathered on this host with vLLM 0.30.0, checking `current_platform.is_pin_memory_available()`:

| Setup | Result |
|---|---|
| vLLM default | False |
| Old InferenceX patch (`vllm.platforms.interface.in_wsl = lambda: False`) | False (no effect) |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | True |
| Worker spawned by vLLM's own mp context, parent imported without `inference_x.api.main` | False |
| Same, parent imported `inference_x.api.main` (runs the new opt-in) | True |
| Same, with `INFERENCE_X_DISABLE_WSL_PIN_MEMORY=1` | False |

The reason the old patch stopped working: 0.30.0's `platforms/cuda.py` does `from .interface import in_wsl`, so replacing the attribute on the `interface` module does not change the name `cuda.py` already holds. It then gates on WSL kernel >= 4.19.121 and `VLLM_WSL2_ENABLE_PIN_MEMORY`.

Decision: keep the probe (a pinned 4 KiB tensor copied to the GPU) and the opt-out, and replace only how the result reaches vLLM. Setting an env var is vLLM's documented mechanism, and child processes inherit it, so the `.pth` hook that re-applied the patch inside every worker is no longer needed. `scripts/install_vllm_patch.sh` is removed. The `.pth` file it had installed into this machine's venv was removed too. Anyone who ran the script before should delete `inferencex_vllm.pth` from their venv's site-packages; left in place it is harmless on 0.30.0 (it only swaps an attribute nothing reads), but it is dead code.

A side observation, kept narrow on purpose: running develop's code on 0.22.1 in a fresh venv with the shipped `.pth` hook installed, vLLM's EngineCore still logged `Using 'pin_memory=False' as WSL is detected`. So the old mechanism was not reliably taking effect in the worker in that setup either. This was not investigated further because the mechanism is being replaced.

`/proc/<pid>/environ` is not usable to check what a vLLM worker inherited: vLLM renames the EngineCore process with setproctitle, which overwrites the start of that memory region. The spawn-based probe above is the evidence instead.

## D4. Dependency changes

`uv lock --upgrade-package vllm`, then `uv sync --all-extras`. InferenceX's own direct dependencies (FastAPI, Pydantic, httpx, prometheus-client, PyYAML, uvicorn) did not move.

- **Changed (30):** vllm 0.22.1 -> 0.30.0; torch 2.11.0 -> 2.13.0; torchvision 0.26.0 -> 0.28.0; transformers 5.10.2 -> 5.15.1; triton 3.6.0 -> 3.7.1; flashinfer-python 0.6.11.post2 -> 0.6.18.post1; compressed-tensors 0.15.0.1 -> 0.17.0; safetensors 0.7.0 -> 0.8.0; fastsafetensors 0.3.2 -> 0.4.0; huggingface-hub 1.16.1 -> 1.33.0; hf-xet 1.5.0 -> 1.6.0; mistral-common 1.11.3 -> 1.12.0; mcp 1.27.2 -> 2.2.0; httpx2 2.4.0 -> 2.13.1; httpcore2 2.4.0 -> 2.13.1; click 8.4.1 -> 8.5.0; apache-tvm-ffi 0.1.9 -> 0.1.11; cuda-toolkit 13.0.2 -> 13.0.3.0; nvidia-cublas 13.1.0.3 -> 13.1.1.3; nvidia-cudnn-cu13 9.19.0.56 -> 9.20.0.48; nvidia-cudnn-frontend 1.18.0 -> 1.30.0; nvidia-cusparselt-cu13 0.8.0 -> 0.8.1; nvidia-nccl-cu13 2.28.9 -> 2.29.7; nvidia-cutlass-dsl, -libs-base and -libs-cu13 4.5.2 -> 4.7.1; humming-kernels 0.1.2 -> 0.1.12; quack-kernels 0.5.0 -> 0.6.5; tilelang 0.1.9 -> 0.1.12; tokenspeed-mla 0.1.2 -> 0.1.8.
- **Added:** instanttensor 0.2.0, mcp-types 2.2.0, nccl4py 0.6.0, nvidia-cuda-nvdisasm 13.4.92, nvidia-cutlass-dsl-libs-core 4.7.1, nvidia-cutlass-dsl-libs-cu12 4.7.1, nvtx 0.2.15, pynvvideocodec 2.0.4, torchcodec 0.16.0.
- **Removed:** diskcache 5.6.3, flashinfer-cubin 0.6.11.post2, gguf 0.19.0, httpx-sse 0.4.3, pyelftools 0.33.

## D5. Qualification matrix

The 0.22.1 comparison runs used develop's code at 863b909 in a separate copy with its own venv resolved from develop's `uv.lock`, on the same card, back to back with the 0.30.0 runs. Both used InferenceX's own server (`uvicorn inference_x.api.main:app`), not vLLM's reference server.

| Check | 0.30.0 | Baseline |
|---|---|---|
| Unit suite | 727 passed, 1 xfailed | green on develop (CI `checks`) |
| ruff, mypy | clean | clean |
| `openspec validate --strict` | pass | |
| Oracle: opt-125m teacher-forced top-1 vs transformers (`make oracle`) | pass | pass (rerun on 0.22.1 in this change) |
| Determinism conformance, one unique sample over N trials (`make verify-determinism`) | pass | pass (rerun on 0.22.1 in this change) |
| Live smoke (`scripts/smoke_test.py`, coder-1.5b) | pass | |
| Aider 0.86.2, `scripts/aider_acceptance.py`, qwen2.5-coder-1.5b | **7/7** (4 streaming, 3 `--no-stream`), 0 HTTP errors | 7/7 on 0.22.1 (EVOLUTION section 15) |
| Continue CLI, `scripts/continue_acceptance.py --model local-4b --trials 10` through `scripts/log_proxy.py` | **10/10**; 50 requests, all 200; 40 turns ended `tool_calls`, 10 ended `stop` | 9/10 on 0.22.1 |
| Manifest `backend_version` | `0.30.0` | |
| Engine timing | present on non-streaming and streaming terminal events; `prefill + decode == inference` on every run | |

The VS Code smoke was not repeated. It was required only if the chat template, the tool parser, or streaming changed materially. The Hermes parser API is unchanged, the chat template is the model's own and did not change, InferenceX's streaming code did not change, and Continue CLI (which exercises the same tool-call stream) passed 10/10.

Continue failures to classify: none occurred.

## D6. Model classes and performance baseline

Warm runs only: two warmup rounds, then five measured runs of each kind. The prompt is 1,147 to 1,168 tokens (depending on the tokenizer), `max_tokens` 256, temperature 0, and every run hit the 256-token limit. Prefill and decode rates come from InferenceX's per-request engine timing (vLLM's own timestamps); TTFT and total latency are measured by the client. Medians are shown. This is a regression check on one card, not a performance claim.

| | coder-1.5b 0.22.1 | coder-1.5b 0.30.0 | 4b-fp8 0.22.1 | 4b-fp8 0.30.0 | 7b-awq 0.30.0 |
|---|---|---|---|---|---|
| Config | bf16, 8192 x 1, util auto (0.6136) | same | FP8, 8192 x 1, util 0.85 | same | AWQ, 4096 x 1, util 0.85 (see below) |
| Linear kernel | | | TritonFp8BlockScaledMM | MarlinFP8ScaledMM | MarlinLinearKernel (AWQ Marlin) |
| Runner | CUDA graphs | CUDA graphs | CUDA graphs | CUDA graphs | CUDA graphs (no eager needed) |
| KV capacity (tokens) | 52,384 | **34,736** | 12,400 | 12,016 | 6,096 |
| Client TTFT (ms) | 126 | 128 | 205 | **318** | 525 |
| Engine prefill (ms) | 120 | 121 | 197 | **308** | 518 |
| Prefill (tok/s) | 9,758 | 9,659 | 5,815 | **3,724** | 2,256 |
| Decode (tok/s) | 88.0 | 88.0 | 50.5 | **58.8** | 60.2 |
| Total, 256 tokens (s) | 3.03 | 3.03 | 5.25 | 4.66 | 4.76 |
| GPU memory used (MiB) | 5,277 | 5,113 | 6,921 | 6,887 | 6,567 |
| Startup, first launch of this config (s) | 69.8 | 69.5 | 68.8 | 70.0 | 70.1 |

Warm restart of coder-1.5b with the compile cache populated: 33.5 s on 0.22.1, 35.5 s on 0.30.0. First launches of a new configuration are about 69 to 70 s on both versions.

**KV capacity at the same `gpu_memory_utilization` dropped.** For coder-1.5b, vLLM reported 1.4 GiB available for KV on 0.22.1 and 0.93 GiB on 0.30.0, with identical weight memory (2.98 GiB). 0.30.0 accounts 0.47 GiB for peak activation in its profiling. Setting `VLLM_USE_V2_MODEL_RUNNER=0` on 0.30.0 gave the same 0.93 GiB (34,864 tokens), so Model Runner V2 is not the cause. The root cause inside vLLM was not isolated. The effect on InferenceX: coder-1.5b still composes to 8192 x 1 with room for 4.24 full-length sequences, and qwen3-4b-fp8 still fits 8192 x 1 (1.47x). No configured context stopped fitting.

**FP8 prefill is slower, decode faster.** 0.30.0 selects the Marlin FP8 kernel for Qwen3-4B-FP8 where 0.22.1 selected the Triton block-scaled kernel. Decode rises 16% and prefill time rises 56%, consistently in every run (0.22.1 prefill 196 to 202 ms; 0.30.0 prefill 307 to 313 ms). For the 256-token workload the total still improves by 11%. For Continue's roughly 4,500-token agent prompts, the extra prefill costs a few hundred milliseconds per turn; Continue trials took 7 to 9 s each and all passed. Overriding vLLM's kernel choice was not attempted.

**7b-awq no longer needs eager mode.** With the shipped config (`gpu_memory_utilization: auto`), 7b-awq fails to start on this card on both versions: InferenceX's auto sizing asks for 0.8995 (7.19 GiB) and vLLM sees 6.93 GiB free. That is a pre-existing auto-sizing issue, not a 0.30.0 regression, and is recorded as a follow-up. To test the model class, a temporary config copy set `gpu_memory_utilization: 0.85` (the value qwen3-4b-fp8 uses); the repo config was not changed. With that:

- 0.22.1 with CUDA graphs: fails. 0.06 GiB left for KV where 0.22 GiB is needed for one 4096-token sequence, and vLLM forces the plain AWQ kernel because the config says `quantization: awq`.
- 0.22.1 with `enforce_eager=True` (vLLM called directly; InferenceX has no eager setting): runs, 4,576 KV tokens.
- 0.30.0 with CUDA graphs: runs, 6,096 KV tokens (1.49x), AWQ Marlin kernel.

So on 0.30.0 the AWQ class works at 4096 context without eager mode and with more KV headroom than 0.22.1 had in eager mode.

**Other observations.** On 0.30.0, Transformers 5.15 logs two `[ERROR]` lines at engine start about undocumented `min_frames`/`max_frames` kwargs on `Qwen3VLVideoProcessorInitKwargs`. They are docstring checks on a video processor InferenceX never uses, and startup continues normally. The first top-p request after startup triggers a Triton JIT compile of vLLM's top-p kernels (logged as a latency spike); the warmup rounds absorb it.

## D7. GO/NO-GO

GO. Every owner criterion held on 0.30.0:

- Correctness: unit suite, oracle top-1 agreement.
- Aider: 7/7, same as 0.22.1.
- Continue tool calling: 10/10 (0.22.1 baseline 9/10).
- Determinism and oracle: both GPU suites pass.
- Metrics and timing truthfulness: manifest KV capacity matches vLLM's own log, engine timing is internally consistent, every native metric InferenceX names is still exported, `backend_version` is correct.
- WSL2 startup: all three model classes start, and pinned memory is now actually enabled in the workers.
- Tested model classes: dense bf16, FP8, and AWQ all serve; AWQ improved (no eager needed).

The two regressions (KV capacity at a fixed utilization, FP8 prefill) are real and recorded, but neither breaks a configured model or an acceptance client.

## Follow-ups (not in this change)

- 7b-awq with `gpu_memory_utilization: auto` cannot start on an 8 GiB card: InferenceX's auto value is computed from free VRAM as seen by the parent process, which is higher than what vLLM sees at engine start. Affects 0.22.1 and 0.30.0 alike.
- Whether to raise `gpu_memory_utilization` for small models to recover the KV capacity 0.30.0 no longer gives at the same setting.
- `scripts/smoke_test.py` defaults to `qwen2.5-0.5b`, so running it against a server that loaded another model returns a generic 400 unless `--model` is passed.
