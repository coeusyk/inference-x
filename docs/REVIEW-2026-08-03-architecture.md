# Inference-X — Deep Architectural Review

**Date:** 2026-08-03
**Reviewer stance:** Staff/Principal engineer, adversarial. Nothing is assumed correct.
**Scope:** systems architecture, inference runtime, long-term product positioning.
**Reference engines studied:** vLLM v0.26.0, llama.cpp b10242, Ollama v0.32.5, SGLang v0.5.16, TGI v3.3.7 (maintenance), Sonar/ex-Aphrodite v0.23.0, Colibrì v1.4.0.
**Downstream consumer evaluated:** Varex v0.1.1 (SPRT prompt evaluation).

---

## 1. Executive assessment

**Inference-X is not an inference engine. It is a control plane over vLLM.** Almost every weakness in the codebase follows from the project not having internalised that distinction.

The evidence is unambiguous. `src/inference_x/engines/vllm_engine.py:263` imports `from vllm import LLM` — the **offline batch class** — and then drives `llm_engine.add_request()` / `llm_engine.step()` manually from a dedicated thread (`engines/driver.py:140,179`). There is no `AsyncLLM` anywhere in the tree. That single decision is the root of a chain of consequences:

- Because `LLM` has no serving loop, the project had to write `EngineDriver` (226 lines) to multiplex concurrent requests. That is a reimplementation of `AsyncLLM`.
- Because `EngineDriver` bypasses vLLM's frontend, the project lost vLLM's `StatLogger` pipeline — which is where vLLM computes queue time, prefill time, decode time, and inter-token latency from the `QUEUED` / `SCHEDULED` / `NEW_TOKENS` events that `EngineCoreOutputs` already carries. So the project had to write its own observability layer, which measures strictly less.
- Because vLLM's scheduler state was no longer visible, the project had to write `AdmissionController` (234 lines) to guess at capacity from outside — reimplementing, with less information, what the scheduler already knows.
- Because `LLM` is not designed to be instantiated more than once per process, multi-model serving required `enforce_eager=True` (no CUDA graphs), a **global cross-engine `step()` lock** (`vllm_engine.py:31`), and a hard clamp of `max_model_len` to 2048. Three performance regressions to support one demo feature.

So roughly 900 lines of the most subtle code in this repo — the driver, the admission controller, the observability middleware, the pool sizing — exist to replace functionality that was **already in the process, already tested by thousands of contributors, and already better.** That is the central architectural finding.

The second finding is a measurement bug that undermines every published number. Throughput is computed as `len(content.split())` — **whitespace-delimited words, not tokens** — in both the live metrics path (`observability/middleware.py:222`) and the benchmark runner (`benchmarks/runner.py:71`). Every `tokens_per_sec` and `mean_throughput_tps` value this project has ever emitted, including the throughput tables in `article-final.md`, is words per second wearing a tokens-per-second label, understating by roughly 25–30% and by a **model-dependent** factor. The advisor then ranks models 40% on that number. The real token count was available and discarded: vLLM returns it in `usage`, but the benchmark runner uses SSE streaming and Inference-X's stream carries no usage chunk.

Relatedly, `suite_version` is documented as "SHA256 of the prompt list" but there is **no `sha256` call anywhere in `src/` or `scripts/`**. It is a hardcoded string in `benchmarks/prompts/standard.json` that nothing verifies. Edit a prompt and every historical result silently becomes incomparable while still claiming the same suite version. For a project whose stated purpose includes reproducible benchmarking, the reproducibility primitive is decorative.

The third finding is strategic, and it is the good news. I researched all six major engines specifically for research-grade reproducibility features. **Not one of them treats run provenance as a first-class output.** `llama-bench` captures build commit and hardware, but only for synthetic microbenchmarks that explicitly exclude tokenization and sampling. SGLang captures per-request parameters to disk, but records nothing about engine identity. vLLM and SGLang both solved *determinism* (`VLLM_BATCH_INVARIANT=1`, `--enable-deterministic-inference`) — but determinism is an **avoidance** strategy: make the answer not depend on batch composition, rather than **record** the batch composition. Nobody records, per request, which other requests shared its forward passes. Nobody emits a run manifest binding engine SHA + model revision + quantization + resolved kernel backend + sampling params to a result.

That is genuinely unclaimed ground, it is a software problem rather than a kernel problem, and it is exactly what Varex needs and does not have. Varex's own archived benchmark manifest records provider and model *labels* and nothing else — no engine version, no model revision, no hardware, no latency, no sampling params (`benchmarks/archive/.../manifest.yaml`), and `benchmarks/baselines/29b400b3.md` explicitly documents the instrumentation gap and warns "do not optimize against this baseline's missing gen/judge split."

**The honest summary:** Inference-X is a competent, unusually well-documented personal platform that is currently losing a fight it should not be in (inference performance, where it is a thin and lossy wrapper around a world-class engine) while ignoring a fight it could win outright (verifiable inference provenance, where the field is empty). The correct move is not to build more inference. It is to stop suppressing vLLM, and to build the provenance layer nobody has built.

---

## 2. Current maturity level

Using a plain 6-level curve:

| Level | Definition | Inference-X |
|---|---|---|
| L0 | Toy script | passed |
| L1 | Working prototype | passed |
| L2 | Personal tool — reliable for its author on its author's machine | **here** |
| L3 | Shareable OSS — a stranger can install, run, and land a PR | blocked |
| L4 | Credible alternative in a defined niche | not attempted |
| L5 | Production infrastructure | out of scope, correctly |

**Verdict: L2.5.** Specifically: *a well-engineered personal platform carrying L4-grade process discipline and L1-grade external compatibility.*

The process discipline is real and unusual. 46 numbered ADRs in `docs/DECISIONS.md` with rationale and supersession chains, an OpenSpec change workflow, ~430 unit tests at a 1.47:1 test-to-source line ratio, and honest post-mortems in `article-final.md`. Most solo projects at this size have none of that. It is the single strongest asset here and it is why the rest of this review is worth writing.

What blocks L3, concretely:

- **No CI.** No `.github/workflows/` at all. 430 tests that only run when the author remembers.
- **No lint or typecheck.** No ruff, no mypy, no black config in `pyproject.toml`. In a 4,700-line Python codebase with `Any`-typed engine internals.
- **No integration tests.** Zero. The `.cursor/rules/backend-python.mdc` rule mandates `unit/`, `integration/`, and `contract/` directories; only `unit/` exists. Every test mocks vLLM, so nothing verifies the system actually works end to end.
- **One hardware configuration.** 6–8 GiB NVIDIA on WSL2. No CPU path, no macOS, no AMD. `vllm>=0.6.0` is a hard dependency and `ModelEntry.engine` is `Literal["vllm"]`, so this is structural, not incidental.
- **API surface too thin to be a drop-in.** See §5.
- **Stale docs.** `CONTRIBUTING.md` claims 286 tests; there are ~430. Two completed OpenSpec changes sit unarchived. `config/models.yaml` documents `gpu_memory_utilization: auto` as `(vram_free - buffer) / vram_total`, which `utils/vllm_pool_config.py:499` explicitly says is *not* what it does.

None of these are hard. All of them are gates.

---

## 3. Biggest architectural weaknesses

Ordered by how much they cost.

### 3.1 Using vLLM's offline `LLM` class instead of `AsyncLLM` — the root cause

`from vllm import LLM` (`vllm_engine.py:263`). `LLM` is vLLM's offline batch entry point. It is not designed to serve concurrent HTTP requests, so everything needed to serve them had to be rebuilt:

| Rebuilt in Inference-X | Already in vLLM's `AsyncLLM` + API server | Cost of the rebuild |
|---|---|---|
| `EngineDriver` thread, `add_request`/`step()` loop, request demux, 20 Hz throttle (226 LOC, DEC-039/043/044) | `AsyncLLM` output processor | A ~40% flaky race (DEC-043), a hand-tuned poll rate, and ongoing maintenance |
| `AdmissionController` KV/context/sequence gating (234 LOC, DEC-038) | vLLM V1 scheduler with `RECOMPUTE` preemption + waiting queue | 429s where vLLM would have queued; fail-open gaps when the tokenizer is unavailable |
| `ObservabilityMiddleware` request timing (285 LOC) | `StatLoggerBase` computing queue/prefill/decode/ITL from engine-core events | The metrics that matter are structurally unreachable (§3.2) |
| Multi-engine VRAM sizing (`vllm_pool_config.py`, 558 LOC) | one process per model | `enforce_eager`, a global step lock, `max_model_len` clamped to 2048 |

This is the highest-leverage fix in the entire review. It deletes code, removes a known race class, and unlocks §3.2 and §8 as side effects.

There is a legitimate counter-argument recorded in DEC-038/039: the author hit real V1 batching bugs and the driver fixed them. Fair. But the fix was to build a parallel serving loop rather than to adopt the one vLLM ships for exactly this purpose, and the review question is whether that is still the right call two vLLM minor versions later. It is not.

### 3.2 The observability layer measures the wrong things, in the wrong units, at the wrong boundary

This is the weakness that most directly blocks the stated research-grade ambition.

**Wrong units.** `len(content.split())` is words, reported as tokens, in both the live path and the benchmark runner. Model-dependent bias of roughly 25–30%. The advisor weights it at 40%.

**Wrong boundary.** TTFT is measured from HTTP request start to the **first SSE body byte** (`middleware.py:245-254`). That number is `queue_time + prefill_time + first_decode + framework_overhead + network`, fused into one scalar with no way to decompose it. vLLM's engine core knows all four separately and Inference-X cannot see any of them, because `EngineDriver` bypassed the layer that surfaces them.

**Wrong things.** Absent entirely: queue time, prefill time, decode time, inter-token latency distribution, preemption count, prefix cache hit rate, scheduler running/waiting depth, batch composition, speculative acceptance rate. There is no Prometheus endpoint. The one genuinely good signal — `kv_capacity_tokens`, read from vLLM internals at init (`vllm_engine.py:395-420`) — is static, captured once at load, and tells you the size of the pool but never its utilisation.

The irony is sharp: **vLLM is already running inside this process and already computes nearly all of it.** vLLM v0.26 exposes `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:inter_token_latency_seconds`, `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries` / `_hits`, `vllm:num_preemptions`, `vllm:num_requests_waiting_by_reason`, and a per-request `metrics` object in the response body behind `--enable-per-request-metrics`. Inference-X ships a strictly worse reimplementation of a subset while the good one sits unwired in the same address space.

**Also unbounded-adjacent:** `JsonLineExporter` (`exporters.py:40-43`) opens, writes, and closes the NDJSON file on **every single request**, synchronously, inside the middleware. That is one `open()`+`fsync`-ish cycle per request on the hot path with no batching and no rotation.

### 3.3 Nothing about determinism or reproducibility exists

`rg '\bseed\b' src/ config/` returns **zero matches.**

- No `seed` in `ChatCompletionRequest`. `SamplingParams` is built from exactly `temperature`, `max_tokens`, `top_p`, and a config `repetition_penalty` (`vllm_engine.py:490-501`).
- `dtype="auto"` is hardcoded (`vllm_engine.py:329`) and not recorded anywhere.
- No HF `revision` pinning. `trust_remote_code=True` is unconditional — both a reproducibility hole (the remote code can change under you) and a genuine security hole.
- vLLM's `VLLM_BATCH_INVARIANT=1` is not set, not surfaced, not mentioned.
- The observability record captures latency and tokens but no sampling fingerprint, no engine version, no model revision.

**Varex cannot use Inference-X today**, and this is why. Varex sends `seed` when configured (`src/models/openai_compat.py:104`); Inference-X's Pydantic model would accept the extra field and silently drop it. A user pinning a seed for reproducibility would get an unpinned run and no warning. That is worse than not supporting seed at all.

### 3.4 `BaseEngine` is a decorative abstraction

`engines/base.py:7-26` declares the right intent in a docstring: *"The rest of the system depends on this interface, not on concrete implementations. Adding a second engine in a future phase must not require changes here."*

It is not true today. Five concrete couplings:

1. `api/deps.py:11,108` imports and constructs `VLLMEngine` directly. No factory, no registry, no `engine:` dispatch.
2. `ModelEntry.engine` is `Literal["vllm"]` — the schema forbids a second backend.
3. `pyproject.toml` lists `vllm>=0.6.0` as a **required** dependency. A CPU-only user cannot `uv sync`.
4. `AdmissionController` reaches for `count_prompt_tokens` and `kv_capacity_tokens` via `getattr` (`admission.py:85`) — methods that are not on `BaseEngine`. A second engine would silently fail open.
5. `utils/cuda_env.py` and `vllm_platform_patch.py` run at **module import time** in `api/main.py:15-16`, before the app object exists. vLLM's platform assumptions are baked into process startup.

The layering above the engine (routes → `ChatService` → `EnginePool`) is genuinely clean and does type against `BaseEngine`. The abstraction is one factory function and a dependency-extra away from being real. But asserting in a docstring that a boundary holds, when the wiring below it hardcodes the implementation, is exactly the kind of comfortable fiction that a review should name.

### 3.5 Multi-model-in-one-process is a demo feature charging architectural rent

To support side-by-side compare in the TUI, the pool path pays:

- `enforce_eager=True` when `pool_size > 1` — CUDA graphs off, measurably slower decode.
- A process-global `threading.Lock` serialising `llm_engine.step()` across all engines (`vllm_engine.py:27-31`).
- `max_model_len` clamped to 2048 regardless of YAML.
- 558 lines of VRAM estimation heuristics with a hand-maintained per-architecture overhead table (DEC-045, DEC-046).

Two OS processes on two ports, with the TUI talking to both over HTTP, would deliver the same feature with none of these costs. The TUI is **already** a pure HTTP client (`playground/` imports no server code), so this is a startup-script change, not a rewrite. On a 6 GiB card two models rarely co-resident anyway — and when they are, the step lock means they are timesharing one forward context, which makes any comparative latency measurement taken in compare mode invalid by construction.

### 3.6 The benchmark subsystem cannot support the claims made of it

Beyond the words-as-tokens bug and the fake `suite_version`:

- **`concurrency` is a stub.** It is accepted, stored in the result JSON, and never used — the runner loops prompts sequentially (`runner.py:159-168`). Every published number is single-request. The admission controller, the driver's batching multiplex, and continuous batching itself are therefore **completely unexercised by the benchmark suite**, despite being the most complex and most bug-prone code in the repo (DEC-038/039/043 were all concurrency bugs found in production, not by tests).
- **`peak_vram_delta_gb` is not a peak and not a delta.** It is `vram_total - min(free_before, free_after)` (`runner.py:119-128`) — occupied VRAM sampled at two instants. It cannot see transient allocation during the run, and it attributes *all* GPU memory including other processes to this model. DEC-032 acknowledges the naming.
- **`quant_score = 1.0  # placeholder`** in the advisor (`advisor.py:138-166`) contributes a constant 10 points to every model. It is 10% of the score doing nothing but rescaling.
- **No warmup, no repetitions, no variance.** One pass, 10 prompts, means and percentiles over 10 samples. `llama-bench` reports per-repetition `samples_ns` and `stddev`; this reports a point estimate with no dispersion. Two runs of the same config will differ and there is no way to know whether a change mattered.

### 3.7 Operationally, there is no lifecycle

Models load eagerly at startup from `INFERENCE_X_LOADED_MODELS` and there is no load/unload API. Changing models means restarting the process. `playground/server_control.py` works around this by shelling out to `pkill -f "uvicorn inference_x"` and `pkill -f "VLLM::EngineCore"` — process management by string matching on command lines, which breaks the moment an invocation changes. `/health` is init-time only (DEC-009), so a wedged engine reports healthy.

### 3.8 Hidden coupling inventory

| # | Coupling | Why it bites |
|---|---|---|
| 1 | `deps.py` → `VLLMEngine` | blocks any second backend |
| 2 | `AdmissionController` → vLLM-only methods via `getattr` | silent fail-open on a new engine |
| 3 | `main.py` import-time CUDA/WSL patching | vLLM assumptions in process startup; untestable |
| 4 | `_POOL_STEP_LOCK` module global | any two engines in a process are coupled forever |
| 5 | `playground/log_feed.py` regex-parses vLLM/uvicorn log lines | UX breaks on an upstream log format change |
| 6 | `server_control.py` `pkill -f` patterns | breaks on invocation change; can kill unrelated processes |
| 7 | Word-count tokenisation duplicated in middleware + runner | same bug, two places, no shared function |
| 8 | `benchmarks/hardware.py` `suggest_gpu_memory_utilization` is dead but tested | `test_hardware.py` locks in behaviour the startup path abandoned (DEC-044) |
| 9 | `textual>=8.2,<9` pinned in main deps | a TUI pin constrains the server package |

### 3.9 Scalability limits (worth stating plainly)

Most do not matter for the stated single-user local scope, but they define the ceiling:

- Single uvicorn worker, no `--workers`; one process, one GPU, no tensor/pipeline parallelism, no multi-node.
- `EngineDriver` submit queue is an **unbounded** `queue.Queue`; the only backpressure is admission control, which is fail-open.
- `InMemoryStorage` is a 1000-record ring buffer — fine, but it means `/v1/metrics` aggregates over "the last 1000 requests," a window that silently varies with traffic.
- NDJSON export is synchronous per-request file I/O on the hot path.
- No auth, no rate limiting (DEC-DEFER-01/02, deliberate), loopback bind only.

---

## 4. Biggest architectural strengths

Not consolation prizes — these are the load-bearing assets the roadmap should be built on.

### 4.1 The decision record is the best thing in this repo

46 ADRs with dates, status, rationale, supersession chains, and — critically — **honest reversals**. DEC-034 superseded by DEC-036. DEC-038 shipped partial, reverted, then fixed by DEC-039. DEC-044 documents that the previous auto-VRAM logic was simply wrong. `article-final.md` publishes the bugs as lessons rather than hiding them.

This matters more than it sounds. It means when this review says "change X," the project can determine *why* X was chosen and whether the reason still holds. Most codebases cannot answer that. It also means the OpenSpec workflow is real rather than ceremonial: archived changes have proposals, spec deltas, and exit criteria.

### 4.2 The layering above the engine is correct

`routes → services → BaseEngine/EnginePool` with routes importing no vLLM, thin handlers, and Pydantic schemas as the contract. `.cursor/rules/api-contracts.mdc` treats schemas as public contracts and requires additive change. The dependency direction is right. The reason §3.4 is a one-week fix rather than a rewrite is that this layering already exists.

### 4.3 The playground is decoupled the right way

`playground/` talks HTTP to `/v1/chat/completions`, `/health`, `/v1/models` and imports zero server code. That is the correct boundary, it validates the OpenAI-compatible contract with a real client on every use, and it means the TUI would work unchanged against `vllm serve`, llama.cpp's `llama-server`, or a two-process split. This is what makes §3.5's fix cheap.

### 4.4 VRAM-tier configuration is a genuinely good idea, better than the upstreams

`config/vram_tiers.yaml` declaring 6/12/24 GiB tiers with `gpu_memory_utilization_ceiling`, `max_model_len_cap`, `max_num_seqs`, `block_size`, `kv_cache_dtype`, `max_num_batched_tokens`, and `enable_prefix_caching`, resolved from probed hardware at startup — is **better ergonomics than vLLM's raw flags.** vLLM makes you know that a 0.5B model on a 6 GiB card wants different `max_num_batched_tokens` than a 7B on a 24 GiB card. Inference-X encodes that knowledge declaratively and applies it automatically.

Same for `variant_selector.py`: `INFERENCE_X_DEFAULT_MODEL=qwen2.5-7b` resolving to the highest-precision variant that fits measured VRAM (bf16 → AWQ) is real, useful, hardware-aware behaviour that no upstream engine does. **Keep both. Build on both.** These are the seeds of the `plan` / `doctor` / `tune` triad in §6.

### 4.5 Failure messages are actionable

`_map_vllm_init_error()`, `preflight_hf_access()` for gated repos, and the loading screen's log parsing (DEC-033) turn multi-minute vLLM stack traces into one-line diagnoses: insufficient VRAM, gated model, access not yet approved. Anyone who has debugged a vLLM OOM on WSL2 knows this is worth real money. It is a differentiator against every upstream.

### 4.6 The test suite, as far as it goes

430 tests, 1.47:1 line ratio, GPU-free by design via `conftest.py` patching `initialize_app`. `test_vllm_pool_config.py` has 23 tests on VRAM arithmetic; `test_engine_driver.py` uses `pytest-repeat` to catch the DEC-043 race. Good instincts. The gap is that there is no CI, no integration layer, and no coverage of the concurrent path under actual concurrency — but the foundation is there.

---

## 5. Comparison against existing inference servers

Two corrections to the premise first, because they change the landscape:

- **TGI is in maintenance mode.** Commit "Maintenance mode (#3345)" landed 2025-12-11; last release v3.3.7, 2025-12-19; the README now redirects users to vLLM, SGLang, llama.cpp, or MLX. Do not benchmark against it as a live competitor.
- **Ollama no longer has its own GGUF engine.** `llm/server.go`: *"All GGML models are served via the upstream llama-server subprocess."* It pins `LLAMA_CPP_VERSION=b10091`. Ollama in 2026 is a Go model-manager, scheduler, and API gateway around llama.cpp, plus an MLX runner for Apple Silicon. So "Ollama vs llama.cpp" is largely one inference core with two operator experiences.

### 5.1 Capability matrix

| | Inference-X | vLLM | llama.cpp | Ollama | SGLang |
|---|---|---|---|---|---|
| Prometheus `/metrics` | **no** | yes (default) | yes (`--metrics`) | **no** | yes (`--enable-metrics`) |
| Per-request queue time | no | **yes** (body + histogram) | no | no | histogram only |
| Prefill/decode split | no | histograms | **yes, in every response** | yes (`prompt_eval_duration`) | histograms |
| Prefix-cache hit signal | no | counters | **per-request `cache_n`** | no | gauge |
| Scheduler depth | no | gauges | `/slots` + 2 gauges | `/api/ps` | **richest** |
| KV cache utilisation | static capacity only | `kv_cache_usage_perc` | no | no | **9+ metrics** |
| Batch-invariant determinism | **no** | `VLLM_BATCH_INVARIANT=1` | **no** (PR #16016 stale) | no | `--enable-deterministic-inference` |
| Per-request `seed` | **no** | yes | yes | yes | yes + `sampling_seed` |
| OpenAI `stop`/`n`/`logprobs` | **no** | yes | yes | yes | yes |
| `stream_options.include_usage` | **no** | yes | yes | yes | yes |
| Tools / JSON mode | no | yes | yes | yes | yes (xgrammar) |
| CPU inference | **no** | limited | **yes** | **yes** | no |
| Speculative decoding | **no** | 8+ methods | extensive | via llama.cpp | MTP/EAGLE |
| CPU/weight offload | **no** | `--cpu-offload-gb`, UVA/prefetch | `-ngl` partial, mmap | auto-fit | limited |
| VRAM-tier auto-config | **yes, unique** | no | `-ngl auto` | auto-fit | no |
| Model-variant-by-VRAM | **yes, unique** | no | no | no | no |
| Actionable OOM diagnosis | **yes, best** | no | no | partial | no |
| Run provenance manifest | no | **no** | `llama-bench` only | no | crash-dump only |

### 5.2 Per-engine reading

**vs vLLM.** Inference-X *is* vLLM, minus its API surface, minus its metrics, minus its determinism, plus a config layer. On raw serving there is no scenario where it wins — it cannot, it is a lossy wrapper. Where it can win: declarative VRAM tiering, variant selection, OOM diagnosis, and the provenance layer vLLM has no interest in building. **Essential:** stop suppressing vLLM (metrics passthrough, `AsyncLLM`, `seed`, batch-invariant flag). **Unnecessary:** anything resembling scheduler or kernel work.

**vs Ollama.** This is the only comparison Inference-X can win convincingly, and the axis is *benchmarking trustworthiness*, not speed. Ollama is actively hostile to reproducible measurement, and each item is verifiable in source: chat history truncation logged at **Debug** level (`server/prompt.go`), so your benchmark can silently evaluate a different prompt than you sent; `num_ctx` clamped to the model's training context with a Warn; **three official docs give three different default context lengths** (2048 in `modelfile.mdx`, 4096 in `faq.mdx`, VRAM-dependent in `context-length.mdx`) so no default can be cited; `keep_alive` defaults to 5m so gaps between runs silently reload the model and reset caches; `OLLAMA_KV_CACHE_TYPE` and `OLLAMA_FLASH_ATTENTION` change numerics and appear nowhere in the response; and `LLAMA_ARG_FIT` defaults on, so the number of offloaded layers is chosen from *currently available* VRAM — two runs on the same machine with a different desktop GPU-memory situation use different `-ngl` and report different numbers with nothing recording which. No Prometheus endpoint at all.

That is the whole product thesis in one paragraph: **Ollama optimises for "it just works"; a benchmarking backend must optimise for "it worked exactly this way, and here is the proof."** Those are different products. **Essential to beat it:** provenance manifest, refusal to silently truncate, explicit config echo. **Unnecessary:** Modelfile, a model-pull registry, a `:latest` tag system.

**vs llama.cpp.** llama.cpp is strictly better on the consumer-hardware axis Inference-X claims (§7): GGUF + mmap + partial `-ngl` offload runs models on a 6 GiB card that vLLM cannot load at all. It also has the **best per-request timing surface of any engine** — `cache_n` vs `prompt_n` vs `predicted_n` with wall-clock for each, in every response, by default, including OpenAI-shaped ones. Its aggregate metrics are the weakest and its determinism story is the worst (PR #16016, explicitly inspired by the Thinking Machines analysis, has been **open and untouched since 2025-09-15**; the maintainer position is *"I don't want to maintain guarantees for bit-for-bit identical results as the batch size is varied"* — the recipe is `--parallel 1` plus prompt caching off). **Essential to learn:** put timings in the response body. **Realistic to become stronger:** be the reproducibility layer llama.cpp has declined to be — over llama.cpp as a backend.

**vs SGLang.** The reference implementation for inference observability and determinism both. ~40 scheduler/KV metrics, per-request `meta_info` with `spec_accept_rate` and `dp_rank`, `--export-metrics-to-file` writing per-request records to disk, `--crash-dump-folder` capturing requests *plus server args plus launch command*, and a **user-runnable determinism conformance suite** (`python3 -m sglang.test.test_deterministic`, pass criterion `Unique samples: 1`). Critically, SGLang's determinism **survives the radix cache** on FA3/Triton backends — vLLM makes no equivalent claim and llama.cpp tells you to disable caching. So SGLang is the one engine where you do not choose between reproducibility and a realistic cache.

Inference-X should treat SGLang as the design target to copy from, not compete with. But note what even SGLang does *not* do: its per-request export records request parameters and nothing about engine identity, and its provenance capture lives on the **crash path**. The instinct is there; the product is not. **Essential to steal:** the conformance-test-as-a-feature pattern, and moving crash-dump-grade provenance onto the happy path.

**vs Sonar (ex-Aphrodite).** `PygmalionAI/aphrodite-engine` now resolves to `dphnAI/sonar`, actively developed (v0.23.0, 2026-07-31), and explicitly a vLLM derivative — `aphrodite/v1/metrics/loggers.py` is a line-for-line mirror of vLLM's with the prefix swapped. Its real remaining differentiator is **samplers** (DRY with 9 parameters, XTC, Mirostat, quadratic smoothing, `top_a`, `tfs`, `typical_p`, `nsigma`, `skew`) which vLLM ships none of. The historical quantization-breadth advantage has largely evaporated; comparing its matrix against vLLM's `QuantizationMethods` literal, the only additions I can identify are `exl3` and `mxfp6`. Relevance to Inference-X: it is the proof that a well-run vLLM fork is a viable project shape — and the proof that "more samplers" is the differentiator available to a fork, which is *not* the one Inference-X should pick, because more samplers means more nondeterminism.

### 5.3 The one-line positioning conclusion

Every engine above competes on **tokens per second per dollar**. Inference-X cannot win there and should stop implying it will. The empty quadrant is **verifiable inference** — and its natural first customer is sitting in the next repo.

---

## 6. Colibri-inspired improvements (only those that genuinely transfer)

Colibrì (JustVugg, Apache 2.0, ~22k stars, v1.4.0) runs GLM-5.2 — 744B MoE, ~40B active — on a 25 GiB consumer box by treating VRAM/RAM/NVMe as one managed hierarchy: ~9.9 GiB of dense int4 weights resident in RAM, 19,456 routed experts (~19 MiB each, ~370 GiB total) streamed from disk on demand with a per-layer LRU, a learned pinned hot-store, and one-layer-ahead router prefetch. Single C file, no BLAS, no Python at runtime.

I want to be blunt about the split, because most of what makes Colibrì famous does not transfer at all.

### 6.1 Does NOT transfer — and why

| Technique | Why it does not apply to Inference-X |
|---|---|
| Expert streaming from disk | Requires **routed MoE sparsity**. Inference-X's registry is dense models 0.125B–7B. A dense model's weights are *all* needed for *every* token; there is no subset to stage. Streaming a dense model from disk is just thrashing. |
| Learned pinned hot-store, `.coli_usage` | Presupposes per-expert routing heat. Dense models have no routing decision to learn. |
| One-layer-ahead router prefetch (`PILOT`) | Predicts *which experts* the next layer needs. Meaningless without experts. |
| Batch-union expert dedup | Deduplicates expert reads across batched positions. No experts, nothing to dedup. |
| Dual-SSD striping, `O_DIRECT`, weighted hash placement | Solves a problem — expert read bandwidth — that dense models on one GPU do not have. |
| MLA compressed KV (576 vs 32,768 floats/token, 57×) | An **architectural property of GLM/DeepSeek attention**, not a runtime option. You cannot add MLA to Qwen2.5; the weights would have to have been trained with it. |
| DSA sparse attention / lightning indexer | GLM-5.2-specific. |
| Single-file C engine, no deps | Actively wrong for this project. Inference-X's value is orchestration and provenance in Python; rewriting a vLLM wrapper in C would delete the only leverage it has. |
| NUMA interleaving | Multi-socket server concern; irrelevant on a laptop. |

**The honest read:** Colibrì's core innovation is *sparsity exploitation*, and Inference-X currently serves no sparse models. Everything below the section on ideas is what remains after removing that.

**The one conditional exception:** if Inference-X ever targets **Qwen3-30B-A3B** (30B total / 3B active) — the realistic MoE model for a consumer box, and already named on Colibrì's own roadmap — then expert-tier placement becomes relevant. That is the single decision that would make Colibrì's core techniques transferable. It is a real strategic option and it should be a deliberate, dated decision rather than a drift. I would not take it in the next two quarters: it means a second engine backend, since vLLM does not do disk-tier expert streaming.

### 6.2 Does transfer — and these are the valuable ones

**(a) "Placement decides speed, never semantics" — as an enforced invariant.**
Colibrì's README: *"there is no SLA on speed, and a hard guarantee on semantics"* and *"Insufficient fast memory may reduce speed; it must not quietly redefine the model."*

This is the single most valuable idea to import, and Inference-X currently **violates it**. `AdmissionController` silently clamps `max_tokens` for `priority: interactive` requests to fit the KV budget. `apply_tier_knobs` silently caps `max_model_len` from a YAML-declared 8192 to a tier's 2048. Multi-engine pools silently clamp to 2048. Every one of those changes what the model computes based on how much memory happens to be free — which is exactly the failure mode Colibrì forbids. And it is fatal for benchmarking: two Varex trials with different concurrency get different effective output limits, and nothing in the response says so.

Concretely: any resolved-vs-requested divergence must be **reported in the response**, not just logged. `{"resolved": {"max_tokens": {"requested": 512, "effective": 384, "reason": "kv_budget_clamp"}}}`. Better: a `strict: true` request flag that returns 400 rather than silently degrading. That is a small change with a large semantic payoff, and it is the thing that makes results trustworthy.

**(b) Token-exact forward validation against a `transformers` oracle.**
Colibrì validates its forward pass teacher-forced against `transformers`, reporting 30–32/32 positions exact and naming the two floating-point near-ties. Inference-X has ~430 tests and **not one asserts that the model produces correct output** — every test mocks vLLM.

A `tests/oracle/` suite that teacher-forces a small model (`facebook/opt-125m`, already in the registry) through both Inference-X and raw `transformers` and asserts top-1 agreement per position would be the first test in this repo that validates *inference* rather than *plumbing*. It is also the natural home for a determinism conformance test, mirroring SGLang's `test_deterministic`.

**(c) `plan` / `doctor` / `tune` — the read-only preflight triad.**
Colibrì ships `coli plan` (inspect planned VRAM/RAM/disk placement), `coli doctor` (read-only readiness check, `--deep` for strict shard/tensor preflight), and `coli tune` (measure and **save** this machine's fastest safe execution profile).

Inference-X already computes almost all of `plan` inside `vllm_pool_config.py` — 558 lines of weight estimation, footprint arithmetic, tier knobs, and pool validation — and then throws it away into log lines. Surfacing it as `make plan` (and `GET /v1/plan`) is nearly free and turns the project's best hidden asset into its most visible feature. `doctor` is the natural home for the already-good error mapping. `tune` is where the benchmark suite finally earns its keep: sweep tier knobs, measure, write the winner to a machine-local profile.

This triad is also the honest answer to "what does Inference-X do that `vllm serve` doesn't."

**(d) The hypothesis table — and the Varex synergy.**
Colibrì's README carries a table of open hypotheses with columns *"evidence so far"* and *"experiment still needed,"* and the contributing guide asks for negative results: *"A well-controlled failure is more valuable here than an unexplained fast number."* Every optimisation is a hypothesis until a controlled end-to-end A/B says otherwise.

This is the same epistemology as Varex, applied to systems instead of prompts. And it closes a loop that is currently open in both repos: **Inference-X's performance claims are exactly the kind of claim Varex's SPRT machinery exists to test.** "Does `kv_cache_dtype: fp8` change output quality?" is an A/B with a statistical stopping rule. "Does ngram speculative decoding preserve outputs?" is a determinism conformance test. Inference-X becomes Varex's backend *and* one of Varex's subjects. That is the strongest reason to pursue this pairing at all, and neither README currently says it.

**(e) The measurement protocol as a data structure.**
Colibrì asks contributors to record *"the hardware, commit, model/container, exact command, prompt, cache state, throughput, TTFT, expert hit rate, bytes read, and quality check; change one variable, repeat the run, and attach raw logs."*

That prose list **is the run manifest schema** from §8. Colibrì enforces it socially, in a CONTRIBUTING file. Inference-X can enforce it mechanically, by emitting it. That is the upgrade, and it is the whole product.

**(f) KV-cache persistence across restarts.**
Colibrì's `.coli_kv` reopens conversations warm with zero re-prefill, *"byte-identical to an uninterrupted session."* Unlike the expert machinery this is **architecture-agnostic** and directly valuable for Varex's workload: 200 trials sharing one long system prompt is a prefix-cache best case. vLLM's APC gives this within a process; persisting across restarts is the increment. Note the reproducibility trap, though — see §8.3 on why cache state must be a recorded field.

**(g) `SPEC_PIN=1` — draft and verify must compute the same function.**
Colibrì's hard-won rule: the MTP head must be int8 (int4 collapses acceptance to 0–4%), and draft and verify must be pinned to one kernel family. Transferable as a *principle* for when speculative decoding lands (§7): a speculative path that changes numerics between draft and verify is a correctness bug, not a tuning knob. Worth writing into the ADR before implementing, not after.

---

## 7. Consumer hardware roadmap

Target: 6–8 GiB NVIDIA laptop (the dev box), widening to 16 GiB desktops and CPU-only.

### 7.1 The structural problem, stated plainly

**vLLM is the wrong engine for a 6 GiB laptop, and this goal is in direct conflict with the current architecture.** vLLM is built for datacenter GPUs: it preallocates a KV pool from a `gpu_memory_utilization` fraction, wants weights resident in VRAM, and has no mmap or partial-offload story worth using. That is why `config/models.yaml` tops out at a 7B AWQ, why the 6 GiB tier caps context at 2048, and why 558 lines of VRAM arithmetic exist at all — they are all symptoms of fighting the engine's assumptions.

llama.cpp does not have this problem: GGUF + mmap + `-ngl N` partial offload runs a 14B on a 6 GiB card, slowly but correctly. So "run larger models on modest hardware" and "vLLM is a required dependency" cannot both be satisfied. **A second backend is the answer to the consumer-hardware goal, and it is the same refactor as §3.4.** That reframing is important: the engine-abstraction fix is not architectural hygiene, it is the enabling step for the stated hardware ambition.

### 7.2 Already implemented — assess as sufficient or not

Per the brief, only flagging where the existing implementation is inadequate.

| Technique | State | Verdict |
|---|---|---|
| Quantization | AWQ plumbed, `quantization` in registry, 2 quantized entries | **Insufficient in coverage, not mechanism.** vLLM supports ~30 methods; the registry uses one. GPTQ/bitsandbytes/fp8 entries are config lines. Also: quantization method is not recorded in any result. |
| Prefix caching | `enable_prefix_caching` per tier | **Insufficient twice over.** (a) **Disabled on the 6 GiB tier** — the exact tier that needs it most, and Varex's repeated-system-prompt workload is the ideal case. (b) No hit-rate metric, so you cannot tell whether it works. Enable it and surface `vllm:prefix_cache_queries`/`_hits`. |
| KV cache quantization | `kv_cache_dtype` plumbed to vLLM | **Plumbed but unused** — `auto` in all three tiers. `fp8` roughly doubles effective context for a ~1 GiB saving at 8192 on a 1.5B model. This is the cheapest context win available: one YAML value. Must be gated behind an oracle quality check (§6.2b) and recorded in the manifest, since it changes numerics. |
| VRAM-aware sizing | 558 LOC, tiers, variant selection | **Sound and a differentiator** — but invisible. Surface as `plan` (§6.2c). |
| Batching / admission | `AdmissionController` + driver multiplex | **Untested under concurrency** — benchmark `concurrency` is a stub. Cannot assess adequacy without measuring it. |
| Scheduler | deferred to vLLM | Correct. Do not touch. |

### 7.3 Not implemented — ranked by payoff per unit of work

**Tier 1 — config-only, do these first**

1. **`--cpu-offload-gb` / `--offload-backend`.** vLLM v0.26 offers UVA zero-copy from pinned host memory and group-based layer prefetch (`vllm/config/offload.py`). This trades PCIe bandwidth for capacity and is the single biggest capacity win available for essentially zero engineering — a registry field and a tier knob. Note it is **weight** offload; KV offload is separate (`--kv-offloading-size`, `--kv-offloading-backend native|lmcache`). Conflating the two is a common error.
2. **`kv_cache_dtype: fp8`** on the 6/12 GiB tiers (see above).
3. **`enable_prefix_caching: true` on the 6 GiB tier**, plus `--prefix-caching-hash-algo sha256_cbor`. That last flag matters more than it looks: the default `sha256` hashes via Python `pickle` and vLLM's own docs state hashes *"may not be reproducible across different Python or vLLM versions."* Comparing cache hit rates across versions without `sha256_cbor` silently compares two different key spaces.

**Tier 2 — real work, high payoff**

4. **n-gram speculative decoding.** vLLM's `ngram` method needs **no draft model and no extra VRAM** — it proposes continuations from the prompt itself. On repetitive, prefix-heavy workloads it is close to free throughput. Varex's workload — one system prompt, 200 trials, structured outputs — is precisely the best case. Gate it behind determinism checks (§6.2g): speculation must be verified output-identical, and vLLM exposes `vllm:spec_decode_num_accepted_tokens` to prove acceptance is paying.
5. **A `llama.cpp` backend behind `BaseEngine`.** This is the real consumer-hardware unlock (mmap, partial offload, CPU-only, GGUF, Metal/Vulkan) and simultaneously proves the abstraction in §3.4. Cheapest credible form: an adapter that proxies a `llama-server` subprocess rather than binding libllama — which also gets you llama.cpp's excellent per-request `timings` block for free.
6. **CPU-only path.** Falls out of (5). Also requires making `vllm` an optional dependency extra, which is a `pyproject.toml` change plus removing the import-time CUDA patching from `main.py`.

**Tier 3 — evaluate, do not assume**

7. **Disk offload / weight streaming beyond vLLM's UVA.** For dense models this is thrashing (§6.1). Only becomes interesting with an MoE target.
8. **MoE expert tiering (Qwen3-30B-A3B).** The Colibrì-shaped bet. High effort, requires a non-vLLM backend, genuinely novel on consumer hardware. A deliberate strategic decision, not a next-quarter item.

### 7.4 Tradeoffs to state explicitly, because they are all quality-relevant

Every technique in Tier 1 and 2 is a **numerics change**, which for a reproducibility-focused project means each one needs the same three things: an oracle quality gate before it ships, a manifest field recording that it was active, and a documented A/B. The tradeoffs:

- **KV fp8:** ~2× context. Measurably degrades long-context recall on some models. Must be recorded — a result computed under fp8 KV is not comparable to one under f16.
- **CPU offload:** capacity for PCIe latency. Decode throughput can fall sharply. Also makes performance sensitive to host RAM pressure, which is *invisible machine state* — the same reproducibility hazard as Ollama's auto-fit.
- **Prefix caching:** large TTFT win on shared prefixes. Introduces cross-request state, so a warm run and a cold run differ. **Cache state becomes a required manifest field.**
- **Speculative decoding:** throughput win when acceptance is high, loss when low. Must be output-identical, and Colibrì's measured 32% *loss* at ~85% expert hit is the cautionary datapoint that this genuinely does not always pay.
- **Quantization:** capacity for quality. Colibrì's own per-row-vs-gs64 int4 finding — ~9pp quality difference, causing think-mode loops and non-terminating generations — is the reference example of quantization *granularity* mattering, not just bit width.

---

## 8. Roadmap to become a research-grade inference server

### 8.1 What "research-grade" means, operationally

An inference server is research-grade when a third party, given only its output artifact, can (a) tell exactly what computation produced a result, (b) determine whether two results are comparable, and (c) reproduce it or know precisely why they cannot.

No engine surveyed does this. That is the opportunity.

### 8.2 The dividing line — in the server vs in an observability layer

The brief asks which of a list belongs where. The test I would apply: **does producing this signal require information only the engine possesses at the moment of computation?** If yes, in-server. If it can be derived from the server's outputs after the fact, outside.

**Must be in the inference server** — the engine is the only witness:

| Capability | Why in-server |
|---|---|
| Queue latency | Only the scheduler knows when a request waited vs prefilled. Unreconstructable from outside; this is the one number a client provably cannot infer. |
| TTFT decomposed (queue / prefill / first-decode) | Requires engine-internal event timestamps. Also: monotonic clocks have per-process reference points, so intervals are only valid between timestamps taken in the same process — which is exactly why vLLM's engine core records `QUEUED`/`SCHEDULED`/`NEW_TOKENS` itself rather than letting the frontend time what it can see. |
| Decode latency / ITL distribution | Per-iteration timestamps. |
| Scheduler metrics, batching statistics | Engine-internal state, sampled per step. |
| KV cache statistics, prefix cache metrics | Allocator-internal. |
| **Batch composition** | The engine knows which requests shared each forward pass and currently every engine discards it. Irrecoverable after the fact. **This is the differentiating one.** |
| Deterministic execution | Kernel-level. Either the engine offers it or nobody can. |
| Reproducible generation | Requires binding sampling params, seed, and engine state at request time. |
| Experiment metadata echo | Only the server knows its own resolved config, engine SHA, model revision, resolved kernel backend. |
| OpenAI compatibility | Definitionally the server's API. |

**Belongs in a separate layer** — derivable from server output:

| Capability | Where, and why |
|---|---|
| Aggregation, percentiles, dashboards | Prometheus/Grafana. Do not reimplement a TSDB. |
| Cross-run comparison, regression detection | The benchmark harness — or Varex, which is a statistical comparison engine and already exists. |
| Benchmark mode | A **client**. The server should not have a benchmark mode; it should expose a `deterministic: true` request flag and honest metrics, and let a harness drive it. Inference-X's runner is already correctly an HTTP client. |
| Replay mode | Split: the server persists request records with full params (SGLang's `--export-metrics-to-file` shape); a **client** replays them. Only the *recording* needs to be in-server. |
| Inference profiling | Server exposes hooks (vLLM already has `--otlp-traces-endpoint` and `--collect-detailed-traces model|worker|all` emitting `gen_ai.latency.time_in_scheduler` / `time_in_model_forward` / `time_in_model_execute`); analysis lives outside. |
| Hardware telemetry | **Ambiguous, and worth being careful.** A *snapshot at request time* must be in-server, because it must be bound to the result. Continuous GPU monitoring is a node exporter's job. Inference-X currently does the snapshot in `benchmarks/hardware.py` — keep that, bind it to the manifest, and do not grow it into a monitoring agent. |

The pattern: **the server's job is to witness and attest. Everything else is a client.** Inference-X has repeatedly built the client-side thing (its own aggregation, its own ranking, its own benchmark runner) while leaving the witness role unbuilt — precisely inverted.

### 8.3 The differentiating artifact: a signed run manifest

The concrete proposal. Every response (and every NDJSON record) carries, or references by content hash, a manifest:

```json
{
  "manifest_version": 1,
  "run_id": "sha256:…",
  "engine":   { "name": "inference-x", "version": "0.3.0", "git_sha": "29b400b3",
                "backend": "vllm", "backend_version": "0.26.0" },
  "model":    { "registry_name": "qwen2.5-7b-awq",
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "hf_revision": "a1b2c3d",
                "weights_sha256": "…",
                "quantization": "awq", "dtype": "float16" },
  "runtime":  { "attention_backend": "FLASH_ATTN", "cuda_graphs": false,
                "enforce_eager": true, "kv_cache_dtype": "fp8",
                "block_size": 16, "max_model_len": 8192,
                "kv_capacity_tokens": 32768,
                "prefix_caching": true, "prefix_cache_hash_algo": "sha256_cbor",
                "batch_invariant": true, "speculative": null },
  "sampling": { "temperature": 0.0, "top_p": 1.0, "seed": 42,
                "max_tokens": 512, "resolved_max_tokens": 512 },
  "request":  { "prompt_sha256": "…", "prompt_tokens": 1041,
                "chat_template_sha256": "…" },
  "timing":   { "queue_ms": 0.4, "prefill_ms": 88.1, "decode_ms": 1240.5,
                "ttft_ms": 88.5, "itl_p50_ms": 9.1, "itl_p95_ms": 11.4 },
  "batch":    { "cold_start": false, "prefix_cache_hit_tokens": 1024,
                "co_batched_request_ids": ["req-7", "req-9"],
                "max_batch_size_observed": 3 },
  "hardware": { "gpu": "NVIDIA GeForce RTX 3060 Laptop", "vram_total_gib": 6.0,
                "driver": "560.35", "cuda": "12.4",
                "cpu": "…", "ram_gib": 32.0, "wsl2": true },
  "warnings": ["max_model_len capped 8192→2048 by tier 6gb"]
}
```

Two fields carry the novelty. **`batch.co_batched_request_ids`** is the variable that determines bitwise output under continuous batching, and no engine records it — vLLM's engine core knows it exactly (it has per-request `SCHEDULED` and per-iteration `NEW_TOKENS`) and discards it after computing histograms. **`warnings`** is the enforcement mechanism for §6.2a: every silent degradation becomes a recorded, machine-readable fact.

`run_id` as a content hash of the manifest makes results **addressable**: two runs are comparable iff their manifests agree on everything outside `timing`. That is a mechanical, checkable definition of comparability, which is what a benchmark corpus needs and what Varex's manifest currently asserts by convention.

Note the honest limit: this is not a *reproducibility guarantee*, it is a *comparability decision procedure*. Bitwise reproduction still requires batch invariance and identical hardware — vLLM's own docs say *"vLLM only provides reproducibility when it runs on the same hardware and the same vLLM version."* The manifest's value is that when a result fails to reproduce, it tells you which field differed. Today no engine can answer that question at all.

### 8.4 Why Varex is the right first customer

Varex's minimum contract is small and Inference-X nearly meets it: non-streaming `POST /v1/chat/completions`, `choices[0].message.content`, `usage.total_tokens`, optional `temperature`/`top_p`/`seed`/`max_tokens`.

**Three blockers today:**

1. **`seed` is dropped silently.** Varex sends it (`src/models/openai_compat.py:104`); Inference-X's schema has no such field. Worst failure mode: user pins a seed, believes the run is pinned, it is not, nothing warns.
2. **429 on admission saturation aborts experiments.** Varex has **no HTTP retries** and stops the whole run after 5 consecutive `ModelClientError`s. Inference-X returns 429 with `Retry-After` for `priority: batch`. A saturated server therefore kills a 200-trial experiment. Either Varex must honour `Retry-After`, or Inference-X must queue rather than reject for a `batch`-priority request. The latter is more correct — batch traffic should wait, not fail.
3. **Silent `max_tokens` clamping** (§6.2a) makes trials incomparable with no record.

**And what Varex gains that nothing else offers.** Varex records provider and model *labels* and nothing more — no engine version, no model revision, no sampling params, no latency, no hardware, no git SHA. Its own `benchmarks/baselines/29b400b3.md` documents the gap and warns against optimising against it. Its README promises a corpus where results "remain reproducible as models change" — but with no runtime provenance, a rerun against a silently updated `llama3.1:latest` is indistinguishable from a real effect. Ollama's model tags are mutable, `keep_alive` and auto-fit vary per run, and none of it is recorded.

So the integration is: **Inference-X emits the manifest; Varex embeds `run_id` and the manifest in `manifest.yaml`.** That single field turns Varex's corpus from "reproducible by convention" into "reproducible by construction," and it is a change to Varex's archive schema plus a new endpoint on Inference-X — no statistics touched.

---

## 9. Prioritized implementation roadmap

Sequenced so each phase is independently shippable and unblocks the next.

### Phase A — Stop the bleeding (1 week, mostly deletions and config)

Nothing here is architectural. All of it is currently producing wrong numbers or wrong behaviour.

| # | Task | Why |
|---|---|---|
| A1 | **Fix token counting.** Support `stream_options.include_usage`, emit a final usage chunk, read real counts. Delete both `len(content.split())` sites. | Every throughput number published is wrong by a model-dependent 25–30%. |
| A2 | **Compute `suite_version`** as SHA256 of the prompt list at load; fail loudly on mismatch with the stored value. | The reproducibility primitive is currently decorative. |
| A3 | **Add `seed`** to the request schema and `SamplingParams`. | Unblocks Varex. One field. |
| A4 | **Add CI** (GitHub Actions: pytest + ruff + mypy). Add ruff/mypy config. | 430 tests that never run automatically. |
| A5 | **Report resolved-vs-requested params** in the response and a `warnings` array. | §6.2a. Stops silent semantic drift. |
| A6 | Delete `quant_score = 1.0`; renormalise advisor weights. Rename `peak_vram_delta_gb` → `vram_footprint_gib` with a compat alias. | Dead weight and a misleading name. |
| A7 | Archive the two completed OpenSpec changes; fix the test count in `CONTRIBUTING.md`; fix the `auto` comment in `models.yaml`. | Housekeeping; cheap credibility. |

### Phase B — Stop suppressing vLLM (2–3 weeks, the highest-leverage phase)

| # | Task | Why |
|---|---|---|
| B1 | **Migrate `LLM` + `EngineDriver` → `AsyncLLM`.** | Root cause fix (§3.1). Deletes ~226 LOC and a race class. |
| B2 | **Expose Prometheus `/metrics`** passing vLLM's stat loggers through. Queue/prefill/decode/ITL/KV/prefix-cache/preemptions arrive for free. | §3.2. Instantly best-in-class-adjacent observability, for near-zero work. |
| B3 | **Per-request timing in the response body** (llama.cpp's `timings` shape), decomposed queue/prefill/decode. | The signal a client cannot compute itself. |
| B4 | **Re-scope admission control** to what the scheduler cannot do; delete the rest. | §3.1. Less code, fewer wrong 429s. |
| B5 | **Queue instead of 429 for `priority: batch`.** | Varex blocker #2. |
| B6 | **Split multi-model into two processes**; delete the global step lock, `enforce_eager` coupling, and the 2048 pool clamp. | §3.5. Removes three perf regressions and ~250 LOC of heuristics. |

### Phase C — The differentiator (3–4 weeks)

| # | Task | Why |
|---|---|---|
| C1 | **Run manifest v1** + `GET /v1/manifest` + `run_id` content hash + `X-Run-Id` on responses. | §8.3. The actual product. |
| C2 | **`batch.co_batched_request_ids`** captured from the engine core. | The one field nobody has. |
| C3 | **`deterministic: true`** request/startup flag wiring `VLLM_BATCH_INVARIANT=1` (+ `VLLM_DETERMINISM_WARMUP_ITERATIONS`, default 3 when on — the first few requests differ due to CUDA graph and JIT warmup); refuse the flag on unsupported hardware (needs SM ≥ 8.0) rather than lying. | Determinism as a contract. |
| C4 | **`tests/oracle/`** — teacher-forced top-1 agreement vs `transformers` on `opt-125m`; plus a runnable determinism conformance test (`make verify-determinism`, pass criterion: 1 unique sample over N trials). | §6.2b. First test that validates inference. Copies SGLang's best pattern. |
| C5 | **Varex integration**: `manifest` block in Varex's archive `manifest.yaml`; document the `provider: custom` config; run one real SPRT experiment against Inference-X end to end. | Proves the thesis with a real consumer. |
| C6 | **`make plan` / `make doctor` / `GET /v1/plan`** surfacing existing sizing logic. | §6.2c. Free feature from existing code. |

### Phase D — Consumer hardware (4–6 weeks)

| # | Task |
|---|---|
| D1 | Tier-1 config wins: `cpu_offload_gb`, `kv_cache_dtype: fp8`, prefix caching on 6 GiB, `prefix_caching_hash_algo: sha256_cbor` — each gated by C4's oracle and recorded in the manifest. |
| D2 | Concurrency in the benchmark runner (make the stub real) + warmup + repetitions + stddev. Then, and only then, assess batching adequacy. |
| D3 | n-gram speculative decoding, gated on output-identity and `spec_decode_num_accepted_tokens`. |
| D4 | `make tune` — sweep tier knobs, persist a machine profile. |
| D5 | Engine factory + `vllm` as an optional extra + a `llama-server`-proxy backend. Unlocks CPU-only, mmap, partial offload, GGUF. |

### Phase E — Only if the thesis proves out

Persistent KV cache across restarts; replay mode; OTel tracing passthrough; MoE expert tiering (the Colibrì bet); a second consuming project.

---

## 10. Things I should absolutely NOT build

Each of these is a plausible-sounding trap, with the reason it is a trap.

1. **Your own inference engine, kernels, or attention implementation.** Colibrì is the cautionary comparison, not the model: it is 22k stars and a large contributor community aimed at MoE disk streaming. You have one 6 GiB laptop. You would be reimplementing vLLM, worse, forever.
2. **A Prometheus/TSDB/dashboard stack.** vLLM's exporter exists; Grafana exists. Passing metrics through is Phase B2. Building a metrics store is not a project, it is a tar pit — and Inference-X's `/v1/metrics` aggregating over "the last 1000 requests" is already a small instance of this mistake.
3. **Multi-model in one process.** Already charging rent (§3.5). Two processes.
4. **Auth, multi-tenancy, rate limiting, API keys.** DEC-DEFER-01/02 got this right. Loopback + a research tool. Do not revisit.
5. **A web dashboard.** Colibrì's Brain and Atlas pages are beautiful and they are *the visualisation of its actual research contribution* (routing heat over 19,456 experts). You have no equivalent to visualise. You have a good TUI. A dashboard would be a screenshot generator.
6. **OpenAI tools / function calling / assistants.** Enormous surface, irrelevant to benchmarking, and every tool-call is a nondeterminism source. `seed`, `stop`, `logprobs`, and `stream_options` are the fields that matter; the rest can 400.
7. **More samplers (DRY, XTC, Mirostat).** This is Sonar's differentiator and it is the exact opposite of yours. Every exotic sampler is another axis of irreproducibility. Deliberately ship *fewer* sampling knobs, well-specified.
8. **Your own quantization pipeline or GGUF converter.** Solved, hard, and orthogonal.
9. **Distributed / multi-node / tensor parallelism.** Contradicts the consumer-hardware thesis. Note also that determinism *across* TP sizes is an open research problem (arXiv:2511.17826 reports 22–63% overhead for the combined approach) — do not walk into it.
10. **A model registry / pull / Modelfile system.** That is Ollama's product. HuggingFace repo + revision is strictly better provenance, which is your whole point.
11. **Fine-tuning, LoRA serving, embeddings, reranking, audio.** Scope creep with no connection to the thesis.
12. **A "benchmark mode" inside the server.** §8.2. The server exposes `deterministic: true` and honest metrics; harnesses drive it. You already got this right; do not regress.
13. **Chasing vLLM on throughput benchmarks.** You *are* vLLM plus overhead. Any such benchmark is a loss by construction, and publishing one invites the comparison you cannot win.

---

## 11. Final recommendation

**Reposition Inference-X from "self-hosted inference server" to "a reproducible inference harness — the attesting control plane over a pluggable engine."**

The one-line pitch: *the only inference server that tells you exactly what computation produced your tokens.*

Three commitments make it real:

1. **Stop competing on inference.** Embrace being a control plane. Adopt `AsyncLLM`, pass vLLM's metrics through, delete the reimplementations. Every line deleted in Phase B makes the project better and faster simultaneously — a rare and worth-taking trade.
2. **Build the run manifest.** It is the only thing here that does not exist anywhere else, it is a software problem you can actually solve on a laptop, and it has a waiting customer.
3. **Make semantics a hard guarantee, speed a best effort.** Colibrì's invariant, imported and enforced: never silently clamp, never silently truncate, never silently swap numerics. Report everything resolved. This is the single behaviour that separates you from Ollama, and Ollama's failure here is well-documented and structural.

**On the maturity question the brief asked** — could this become a research-grade inference server rather than another Ollama clone? Yes, but not by becoming a better Ollama. It becomes research-grade by becoming a *different kind of thing*: not the engine, the instrument attached to it. That is a smaller product than "an inference server," it is achievable by one person on a 6 GiB laptop, and it is unoccupied.

**The main risk is honest and worth naming:** this is a bet that reproducible local inference is a real need rather than a personal one. The evidence for it is thin — my survey found only nascent, low-adoption provenance tooling (`verievals`, `sandboxledger`, RS-1) and none of it inside an engine. Varex is the mitigation: a real consumer with a real documented gap, in your own hands. If the manifest does not visibly improve Varex's corpus within one integration, the thesis is weaker than it looks and you should know that in weeks rather than quarters. Phase C5 exists to force that test early.

**Secondary risks:** the `AsyncLLM` migration could surface new batching bugs (mitigation: Phase A4's CI plus C4's oracle tests land first — currently there is nothing that would catch a regression); vLLM's V1 internals shift between minor versions and B2/C2 depend on them (mitigation: pin vLLM, record its version in the manifest — which is exactly what the manifest is for); and the biggest risk of all is Phase A/B feeling like janitorial work and getting skipped for the exciting Phase C. It cannot be. Building a provenance system on top of a throughput metric that counts words as tokens would be recording, precisely and verifiably, the wrong number.

---

## Immediate Next Sprint

**Goal:** every number Inference-X reports is correct, and Varex can talk to it.
**Duration:** 1 week. **Exit criterion:** one real Varex SPRT experiment completes against Inference-X with a pinned seed, and every reported throughput figure is in real tokens.

Follow the OpenSpec workflow — this is a contract change, so it needs `openspec/changes/2026-08-04-truthful-metrics-and-seed/` with proposal, spec deltas, and tasks before code.

**Task list**

1. **CI first** (`.github/workflows/ci.yml`): `uv sync`, `pytest tests/unit`, `ruff check`, `mypy src/`. Add `[tool.ruff]` and `[tool.mypy]` to `pyproject.toml`. Expect an initial mypy baseline with `ignore_errors` on the vLLM-touching modules; do not block on a clean run. *Rationale: nothing else in this sprint is safe without it.*

2. **Real token counts.**
   - Add `stream_options: {include_usage: bool}` to `ChatCompletionRequest`.
   - Emit a final SSE chunk carrying `usage` before `[DONE]`, and a `finish_reason` on the last content chunk (both currently missing — see `services/chat_service.py:94-107`).
   - Delete `_count_sse_delta_tokens` (`observability/middleware.py:203-224`) and the `len(content.split())` in `benchmarks/runner.py:71`; read `usage.completion_tokens` instead.
   - Tests: assert a streamed request's `completion_tokens` equals the non-streamed count for the same prompt.

3. **`seed` end to end.**
   - `seed: Optional[int]` on `ChatCompletionRequest`; thread into `_sampling_params()` (`vllm_engine.py:490`).
   - Echo the effective seed in the response.
   - Test: same seed + `temperature=0` twice → identical output, single-request. Mark it `xfail` under concurrency and reference Phase C3 — do not overclaim.

4. **`suite_version` computed, not asserted.** SHA256 over the canonical JSON of the prompt list at load. If it disagrees with the stored value, raise. Add `make suite-version` to regenerate. Regenerate `standard.json`'s value and note in the ADR that pre-existing results are now correctly marked incomparable.

5. **Truthful resolution.** Add `warnings: list[str]` and a `resolved` block to `ChatCompletionResponse`. Populate from `AdmissionController` clamps and `apply_tier_knobs` caps. Add `strict: bool = false`; when true, return 400 instead of clamping. Test both paths.

6. **Varex smoke test.** In Varex, a config with `{"protocol": "openai_compat", "provider": "custom", "base_url": "http://localhost:8000/v1", "model": "qwen2.5-0.5b", "temperature": 0.0, "seed": 42}`. Run `configs/demo_qa_objective.json` against it. Log wall-clock. **Expect a 429 abort under load** — that is Phase B5's evidence, so record it rather than patching around it.

7. **ADR + docs.** New DEC entries for: token counting was wrong and all prior throughput numbers are superseded (be explicit — `article-final.md` needs a correction note); seed support; strict mode. Update `PHASES.md` or, better, record it as an OpenSpec change and stop adding phases.

**Explicitly out of scope this sprint:** `AsyncLLM` (Phase B), Prometheus (B2), the manifest (C1), any consumer-hardware knob (D). Resist all four.

---

## Note to Future Me

Paste this into a fresh conversation to continue designing Inference-X.

> **Context.** I'm designing Inference-X (`~/projects/inference-x`, MIT, Python 3.13, FastAPI + vLLM, ~4,700 LOC src, ~430 unit tests, 46 ADRs in `docs/DECISIONS.md`, OpenSpec workflow). Dev box: 6–8 GiB NVIDIA laptop on WSL2. A full architecture review lives at `docs/REVIEW-2026-08-03-architecture.md`. Downstream consumer: Varex (`~/projects/varex`), an SPRT prompt-A/B tool that needs a trustworthy backend.
>
> **The decision I made.** Inference-X is **not** an inference engine — it's a *control plane over a pluggable engine*, and its product is **verifiable inference**: the only server that tells you exactly what computation produced your tokens. I am not competing with vLLM on throughput (I *am* vLLM plus overhead) and not competing with Ollama on convenience. Positioning: **research/benchmarking backend**, with a replacement backend for Varex as the first proof.
>
> **Why that's defensible.** I surveyed vLLM 0.26, llama.cpp b10242, Ollama 0.32.5, SGLang 0.5.16, TGI 3.3.7, Sonar 0.23.0. **No engine treats run provenance as a first-class output.** `llama-bench` records build+hardware but only for synthetic microbenchmarks. SGLang exports per-request params but nothing about engine identity, and its best provenance capture (server args + launch command) is on the *crash* path. vLLM and SGLang solved *determinism* (`VLLM_BATCH_INVARIANT=1`, `--enable-deterministic-inference`) — but that's **avoidance** (make output not depend on batch composition), not **observability** (record it). Nobody records which requests were co-batched. That's my gap.
>
> **Three facts that reframe the landscape:** TGI went into maintenance mode 2025-12-11 and redirects to vLLM/SGLang. Ollama no longer has its own GGUF engine — `llm/server.go` shells out to upstream `llama-server` (pinned b10091). SGLang's determinism uniquely **survives the prefix cache** (FA3/Triton), so it's the reference design.
>
> **The root architectural flaw.** `vllm_engine.py:263` imports vLLM's **offline `LLM`** class, then drives `llm_engine.add_request/step()` from a hand-written `EngineDriver` thread. No `AsyncLLM` anywhere. Consequence: I reimplemented — worse — the serving loop, admission control, and observability that were already in the same process. It also cost me vLLM's `StatLogger` pipeline, which is why I have no queue/prefill/decode metrics. Multi-model-in-one-process makes it worse: global `step()` lock, `enforce_eager`, `max_model_len` clamped to 2048. **Fix: migrate to `AsyncLLM`, pass vLLM's Prometheus metrics through, split multi-model into two processes.** Phase B deletes ~500 LOC and makes it faster.
>
> **Two bugs that invalidate published numbers.** (1) Throughput is `len(content.split())` — **words counted as tokens** — in both `observability/middleware.py:222` and `benchmarks/runner.py:71`. Every tok/s figure I've published, including in `article-final.md`, is wrong by a model-dependent 25–30%, and the advisor weights it 40%. Real counts are available in vLLM's `usage`; the SSE stream discards them because I don't support `stream_options.include_usage`. (2) `suite_version` is documented as SHA256 of the prompt list but **no `sha256` call exists anywhere in the repo** — it's a hardcoded string nothing verifies.
>
> **Varex blockers:** no `seed` in my schema (Varex sends it; I silently drop it — worse than not supporting it); 429 on admission saturation kills experiments because Varex has no retries and aborts after 5 consecutive errors; silent `max_tokens` clamping makes trials incomparable with no record.
>
> **The differentiating artifact: a signed run manifest** on every response — engine git SHA + backend version, HF repo + **revision** + weights hash + quantization + dtype, resolved attention backend / CUDA-graph mode / `kv_cache_dtype` / prefix-caching config, sampling params incl. seed, decomposed queue/prefill/decode timings, hardware snapshot, a `warnings[]` array for every silent degradation, and — the novel field — **`batch.co_batched_request_ids`**. `run_id` = content hash of the manifest, so two runs are comparable iff their manifests agree outside `timing`. Varex embeds `run_id` in its archive manifest, turning its corpus from reproducible-by-convention into reproducible-by-construction. Be honest that this is a *comparability decision procedure*, not a reproducibility guarantee.
>
> **Colibrì (JustVugg, GLM-5.2 744B MoE on 25 GiB): what transfers.** NOT the famous parts — expert disk streaming, learned pin sets, PILOT prefetch, dual-SSD striping, MLA compressed KV, single-file C — all require **routed MoE sparsity** or are architecture-baked, and my registry is dense 0.125B–7B models. What DOES transfer: (a) the invariant **"placement decides speed, never semantics"** — I currently violate it via silent clamps; (b) **token-exact forward validation against a `transformers` oracle** — I have 430 tests and not one checks model correctness; (c) **`plan` / `doctor` / `tune`** read-only preflight commands — I already compute `plan` in 558 LOC of `vllm_pool_config.py` and throw it into log lines; (d) the **hypothesis table + negative-results culture** — and note the loop this closes: *my performance claims are exactly what Varex's SPRT exists to test*; (e) Colibrì's contributor measurement checklist literally *is* my manifest schema; (f) **KV persistence across restarts** (architecture-agnostic, and Varex's 200-trials-one-system-prompt workload is the ideal case); (g) `SPEC_PIN=1` — draft and verify must compute the same function.
>
> **Consumer hardware — the structural tension.** vLLM is the wrong engine for a 6 GiB laptop (preallocated KV pool, no mmap, no real partial offload); llama.cpp runs a 14B there. So "larger models on modest hardware" and "vLLM as a required dependency" are incompatible — **a second backend is the answer to the hardware goal, and it's the same refactor as fixing my fake `BaseEngine` abstraction** (`deps.py:11,108` hardcodes `VLLMEngine`; `ModelEntry.engine` is `Literal["vllm"]`; `vllm` is a required dep). Cheapest form: an adapter proxying a `llama-server` subprocess, which also gets me llama.cpp's excellent per-request `timings` block free. Config-only wins first: `--cpu-offload-gb`/`--offload-backend` (UVA/prefetch — note this is *weight* offload, KV offload is the separate `--kv-offloading-size`), `kv_cache_dtype: fp8` (plumbed but set to `auto` in all tiers), prefix caching **on the 6 GiB tier where it's currently off**, and `--prefix-caching-hash-algo sha256_cbor` (the default `sha256` pickles and vLLM's docs say hashes may not be reproducible across Python/vLLM versions). Then n-gram speculative decoding — no draft model, no extra VRAM, and my repetitive-prefix workload is its best case. Every one of these changes numerics, so each needs an oracle gate + a manifest field + a documented A/B.
>
> **What I keep** (genuine differentiators no upstream has): declarative VRAM tiers (`config/vram_tiers.yaml`), model-variant-by-measured-VRAM (`variant_selector.py`), and actionable OOM/gated-model diagnosis. **What I must not build:** my own engine or kernels; a metrics store or dashboard; multi-model in one process; auth/multi-tenancy; tools/function calling; more samplers (that's Sonar's differentiator and the *opposite* of mine — every exotic sampler is another irreproducibility axis); a Modelfile/pull registry; distributed/TP; a "benchmark mode" *in* the server (the server attests, clients benchmark).
>
> **Server vs external, my dividing line:** in-server iff the engine is the only witness at the moment of computation — queue latency, decomposed TTFT, ITL, scheduler/batch/KV/prefix-cache state, **batch composition**, determinism, and the config/provenance echo. External — aggregation, dashboards, cross-run comparison (that's Varex), replay driving, profiling analysis. Hardware telemetry splits: a request-time *snapshot* is in-server because it must bind to the result; continuous monitoring is a node exporter's job.
>
> **Roadmap.** A (1wk): CI + ruff/mypy; real token counts via `stream_options.include_usage`; `seed`; computed `suite_version`; `warnings[]` + `resolved` block + `strict: true`; drop the dead `quant_score` placeholder. B (2–3wk): `AsyncLLM`; Prometheus passthrough; per-request timings in the body; re-scope admission control; queue instead of 429 for batch priority; two-process multi-model. C (3–4wk): run manifest v1; `co_batched_request_ids`; `deterministic: true` → `VLLM_BATCH_INVARIANT=1` (needs SM≥8.0, plus warmup iterations — refuse rather than lie on unsupported hardware); `tests/oracle/` + `make verify-determinism`; **Varex end-to-end integration**; `make plan`/`doctor`. D (4–6wk): the config-only hardware wins; real concurrency + warmup + stddev in the benchmark runner; n-gram spec decode; `make tune`; engine factory + llama.cpp backend.
>
> **Where I am right now:** starting Phase A. Next concrete step is the OpenSpec change `2026-08-04-truthful-metrics-and-seed` — CI first, then real token counting, then `seed`. **The trap to avoid is skipping A/B to get to the exciting manifest work in C:** building a provenance system on top of a metric that counts words as tokens would be recording the wrong number, precisely and verifiably.
>
> **Falsification condition:** this whole thesis bets that reproducible local inference is a real need, and the external evidence is thin (only nascent, low-adoption tooling — `verievals`, `sandboxledger`, RS-1 — and none of it inside an engine). Varex is the test. If the manifest doesn't visibly improve Varex's benchmark corpus within one integration, the thesis is weaker than it looks. Phase C5 exists to find that out in weeks, not quarters.
