# Phase A Architecture Reference

This document describes the Inference-X runtime as it exists after OS-6, the
final Phase A unit. Phase A ran as OS-1 through OS-6, each landed as an
accepted ADR in `docs/DECISIONS.md`. This document is a stable entry point
into that record — it summarizes the accepted architecture; it does not
restate ADR text, and it settles nothing that DEC-047 through DEC-057 did not
already settle. When this document and an ADR appear to disagree, the ADR
wins (see Document authority in `docs/ARCHITECTURE.md`).

> **Freeze.** This document describes the runtime at the completion of Phase A.
> Phase B changes should be documented in a separate `PHASE-B-ARCHITECTURE.md`
> (or equivalent) rather than rewriting this document in place. Treat this file
> as a historical architectural baseline once Phase B begins.

## Current Runtime Snapshot

| Property           | Current value                   |
| ------------------ | ------------------------------- |
| Runtime            | Offline vLLM `LLM` + `EngineDriver` |
| Streaming          | `ChatStreamChunk`               |
| Resolution         | Effective Request (`resolved`)  |
| Determinism        | Seed honoured only              |
| Benchmark identity | SHA-256 `suite_version`         |
| Advisor            | Measured-only score             |
| Backend count      | 1 (vLLM)                        |
| AsyncLLM           | Not yet                         |
| Prometheus         | Phase B                         |
| OpenTelemetry      | Not yet                         |

## Non-goals of this document

This document intentionally does not describe:

- AsyncLLM
- Scheduler internals
- Multi-backend execution
- Queue architecture
- Prometheus metrics
- OpenTelemetry traces
- Batch determinism
- Engine factory implementation

Those items appear only as deferred assumptions or roadmap pointers (§10).
They are not part of the Phase A architecture described here.

## 1. Runtime architecture

Inference-X is a self-hosted, OpenAI-compatible inference runtime. It is not
described as "a vLLM application" — the runtime owns architectural and
execution policy, and inference backends are replaceable implementation
details over the project's lifetime (DEC-047).

Layered dependency direction: `API -> Services -> Interfaces -> Implementations`.

```text
API
 ↓
Services
 ↓
BaseEngine
 ↓
VLLMEngine
 ↓
EngineDriver
 ↓
vLLM
```

Benchmark path (separate from the request path; storage filters before the
advisor ranks):

```text
BenchmarkRunner
        ↓
SuiteIdentity
        ↓
Storage
        ↓
Advisor
```

```text
src/inference_x/
├── api/            HTTP layer: routes, deps (composition root), errors
├── schemas/        Typed request/response models (chat, model, metrics, common)
├── services/        Use-case orchestration (chat_service, metrics_service, model_service)
├── engines/         BaseEngine contract, registry (empty), VLLMEngine, driver, pool
├── routing/         Admission, policies, task routing, variant selection
├── observability/   Middleware, recorder, storage, exporters
├── benchmarks/       Hardware profiling, runner, storage, advisor, suite_identity
├── core/             Settings, lifecycle
└── utils/            cuda_env, ids, vllm_platform_patch, vllm_pool_config, vram_tiers
```

Full layer/module responsibilities are in `docs/ARCHITECTURE.md`; this
document does not repeat them, only what Phase A changed inside that shape.

Today the runtime serves inference through vLLM's offline `LLM` class plus a
shared `EngineDriver` thread (DEC-038/DEC-039), not `AsyncLLM`. That
migration is Phase B, not Phase A (see §10).

## 2. Engine Boundary (DEC-047)

`BaseEngine` (`engines/base.py`) is the sole contract the rest of the system
depends on. As of OS-6:

- Speaks `schemas.chat` wire types only — no `inference_x/execution/`
  package and no second (execution-level) type system exist or are
  authorized.
- Declares one durable capability: `count_prompt_tokens(request) -> int |
  None`, called directly by admission — no `getattr` discovery for it.
- `kv_capacity_tokens` remains provisional and undeclared on the contract;
  callers still discover it via `getattr`. Its fate is decided at Phase B4
  (admission rescope), not here.
- `engines/registry.py` is still empty, and `api/deps.py` still constructs
  `VLLMEngine(...)` directly. The factory (DEC-047 exit criterion 3) is
  accepted hygiene, not yet built — it is nonblocking, parallelizable work
  that must not gate Phase B (AsyncLLM).
- vLLM remains a required dependency (DEC-007, untouched). Optional-`vllm`
  and a second concrete backend are both explicit non-decisions of DEC-047.

Backend plurality is a long-term architectural direction, not a delivery
commitment: backend-neutral abstractions are not introduced until justified
by a second concrete implementation ("rule of two").

## 3. Streaming protocol (DEC-049, DEC-053)

`BaseEngine.generate_stream` yields `ChatStreamChunk` (`schemas/chat.py`), not
`str`:

- `content: str` — delta text, empty on non-content events.
- `finish_reason: Literal["stop", "length", "error"] | None` — set only on
  the terminal event.
- `usage: ChatCompletionUsage | None` — set on the terminal event when the
  backend can account it; never estimated.

Wire-order rules, fixed for the life of this contract absent a new DEC:

- **Pre-generation phase** ends at the first sampled token and contains
  **exactly one** event. A field may appear in it only if fully determined
  and immutable at admission time (e.g. `resolved`, `warnings`) — a fact
  that can change during generation must not appear here.
- **Post-generation phase**: the usage event is the last event before
  `data: [DONE]`. Nothing is emitted after it.
- `resolved` (Effective Request, §5) is a subset of what the prologue phase
  is allowed to carry, not the definition of the prologue — a later phase
  must not widen `resolved` just because something is prologue-eligible.
- TTFT is measured from the first **content** event, not the first SSE chunk.
- `stream_options.include_usage` (OpenAI-compatible, `schemas/chat.py`)
  gates whether the terminal usage chunk is emitted at all.

Streamed and non-streamed paths share one `_sampling_params` builder and one
`derive_terminal_metadata` helper so usage cannot drift between them
(DEC-050).

## 4. Runtime Resolution — the Effective Request (DEC-052, DEC-053)

- `resolved: ResolvedRequest | None` and `warnings: list[ResponseWarning]`
  are surfaced on both `ChatCompletionResponse` and, per §3, the streamed
  prologue event.
- `resolved` contains every field that exists on `ChatCompletionRequest`
  minus `messages` and transport/policy controls — derivability, not
  determinacy, governs its membership.
- `ResponseWarning.type` is `"substituted"` (a value was silently clamped)
  or `"degraded"` (a check was skipped, no value changed). The closed
  `(type, code, field)` registry lives in DEC-053; adding a code is a spec
  change.
- **`strict` invariant (DEC-052):** `strict` may only convert a
  substitution into a rejection — it can never change what would have been
  substituted. Under any request accepted in both modes with an identical
  seed, completion content is byte-identical. Phase B/C runtime-policy needs
  (batch isolation, cold-cache fail, revision pin) get their own fields and
  must never be folded into `strict`.

## 5. Deterministic Generation (DEC-051)

- `seed: Optional[int] = None` on `ChatCompletionRequest`, forwarded to
  `SamplingParams` unchanged when set — no rewrite, clamp, or
  normalization, including `-1`.
- Omitting `seed` reproduces pre-OS-3 sampling construction exactly.
- Streaming and non-streaming share the same sampling-params builder.
- Explicitly **not** guaranteed by this contract: cross-request/batch
  identity, cross-hardware or cross-version identity, CUDA-graph/JIT/
  prefix-cache/speculative-decode identity, replay, response seed echo, or
  any interpretation of backend-specific sentinel values. "Honoured" means
  the value reaches the sampler — not that the server is deterministic.
  Batch-invariant determinism is Phase C3.

## 6. Benchmark identity (DEC-054, DEC-055)

- `benchmarks/suite_identity.py` is the **sole** canonicalizer.
  `suite_version = sha256(json.dumps(prompts, sort_keys=True,
  ensure_ascii=False, separators=(",", ":")))`, hashed over
  `data["prompts"]` only (parsed in-memory form — file formatting is
  excluded from identity). Stored as bare hex.
- Suite load fails loudly (`SuiteIdentityError`) on a missing or mismatched
  `suite_version`, naming `make suite-version` as the fix. This is
  verification of an existing pinned literal, not a migration — all
  historical results remain comparable and are never regenerated by this
  change.
- `storage.py:latest_per_model_for_suite(expected_suite_version)` filters
  results to the expected suite **before** latest-per-model selection.
  Mixed-suite result sets never reach the advisor. "No results" and
  "results exist but none match this suite" are distinguishable outcomes.
- `suite_version` is a **necessary, not sufficient**, comparability key: it
  pins the benchmark input, not the runtime version or concurrency that
  produced the numbers. See §10 for the coupling this leaves open.

## 7. Advisor semantics (DEC-056, DEC-057)

- `benchmarks/advisor.py` scores three measured quantities only:
  throughput (weight `4/9`), warm TTFT (`1/3`), VRAM headroom (`2/9`).
  `quant_score` (a constant `1.0`) is deleted — it never discriminated
  between models, only inflated every score by a fixed floor. Weights are
  stated as uncalibrated editorial preference, not a fitted signal.
- `viable` is the only viability signal. A VRAM-gated model and a
  viable-but-worst model can both show `score = 0.0`; only `viable`
  distinguishes them. No consumer may infer viability from the score.
- `score` is a within-report ordinal: valid only for ranking viable models
  from the same benchmark suite, on the same hardware, under the current
  weights. Not portable across suites, hardware, or future weight changes.
- The VRAM field is `vram_device_occupied_gib` on `BenchmarkResult`
  (renamed from `peak_vram_delta_gb`, which named a "delta" while measuring
  `total − min(free)` device occupancy). `peak_vram_delta_gb`
  (`BenchmarkResult`) and `vram_gb` (`AdvisorResult`) remain readable
  deprecated aliases through Phase A; the canonical field always wins on
  conflict; removing either alias requires a future ADR.

## 8. Compatibility guarantees

- **Streaming**: additive only. Unknown chunk fields, the terminal usage
  chunk, and `stream_options` are all safely ignored by clients built
  against the pre-DEC-049 contract.
- **Benchmark JSON**: read-only under Phase A. Historical result files load
  via deprecated aliases and are never rewritten; new results write the
  canonical field name.
- **`strict`/`resolved`/`warnings`**: additive surfaces. Default (absent
  `strict`) behavior is unchanged aside from these additive fields.
- **Metrics**: pre-OS-2 streamed token counts (word-count approximations)
  and post-OS-2 counts (real usage-derived counts) are declared
  **permanently incomparable** (DEC-050) — never mix them in one series.
  Non-streamed `usage` was always engine-accounted and is unaffected.
- **Lint/type-check gate (DEC-048)**: `ruff check .` runs `E4/E7/E9/F` only;
  `mypy src/` carries a closed, enumerated 4-module baseline
  (`engines/vllm_engine`, `engines/driver`, `services/chat_service`,
  `api/deps`) that must not grow during Phase A. `engines/driver`'s entry is
  expected to disappear with the module in Phase B1.

## 9. Repository ownership

Per DEC-047, ownership boundaries for the Engine Boundary and its
neighbors:

| Component | Owns | Knows | Must never know |
|---|---|---|---|
| `BaseEngine` | Generation façade, declared capabilities | Current wire request/response types; capability `None` semantics | HTTP/FastAPI; other backends; admission policy; model selection |
| `engines/registry` | Instantiation dispatch by engine type (not yet wired into `api/deps.py`) | Registered backend constructors; model config needed to construct | Admission; routing policy; OpenAI wire minting |
| `routing/` (admission) | Whether a request may run | Declared engine capabilities; registry model limits; priority semantics | vLLM internals; CUDA; the Driver/`step()` loop |
| `api/` + `services/` | Wire validation; composition-root wiring; OpenAI response minting | Settings, pool, router, admission | Backend-native serving loops |
| `VLLMEngine` + `EngineDriver` | Inference execution for vLLM | vLLM APIs; local runtime state; translation of current request types to native calls | HTTP schemas as owned long-term vocabulary (today borrowed); other backends; global scheduling policy |
| `benchmarks/suite_identity` | Suite content-hash canonicalization | Parsed prompt collection | Advisor scoring; storage filtering |
| `benchmarks/storage` | Result persistence, suite-aware selection | `suite_version` per result | Advisor weighting |
| `benchmarks/advisor` | Scoring and ranking | Measured throughput/TTFT/VRAM; `viable` | Suite-hash verification; storage filtering |

## 10. Deferred Phase B assumptions

Two phase-numbering schemes coexist in this repository and are not the same
scheme: `docs/PHASES.md` numbers phases 0–12 (chronological delivery
history); `docs/REVIEW-2026-08-03-architecture.md` §9 separately numbers
Phase A–E (the roadmap DEC-047 onward implements, where "Phase A" is OS-1
through OS-6 and this document). "Phase B1", "Phase C3", etc. below refer
to the A–E scheme, not to `PHASES.md`.

Phase A intentionally leaves the following as recorded, unaddressed
assumptions in the current code — not defects, but facts a future runtime
change must recognize as invalidated rather than treat as unrelated:

- `_warm_ttft_ms` drops `prompt_results[0]` as a cold-start artifact,
  assuming the first request is the sole anomaly and later ones are
  independent. Under a runtime where TTFT depends on queue occupancy
  (AsyncLLM), "warm" stops meaning what the name says.
- `BenchmarkResult.concurrency` is recorded and never read.
  `mean_throughput_tps` is a single scalar with no concurrency qualifier,
  weighted at 44% by the advisor.
- `vram_device_occupied_gib` attributes whole-device VRAM occupancy to one
  model. With one resident model this is a loose upper bound; with more
  than one, it is not attributable at all — yet it drives the hard
  viability gate.
- `suite_version` pins the benchmark input, not the runtime version that
  produced the numbers. Results with identical suite hashes can still be
  runtime-incomparable.
- `runner._check_vram_budget` compares a measured device-occupancy number
  against a static analytic estimate — two different quantities compared
  under one gate, whose assumptions must be kept in sync by hand.

Broader Phase B/C roadmap (`docs/REVIEW-2026-08-03-architecture.md` §9),
sequenced after Phase A and explicitly not gated by Engine Boundary hygiene:

- **Phase B** (highest leverage): migrate `LLM` + `EngineDriver` to
  `AsyncLLM` (**B1 — complete**, deletes the driver and its DEC-038/039
  race-class history); expose vLLM's native Prometheus stats (**B2 —
  complete**); per-request queue/prefill/decode timing in the response
  body (**B3 — complete**);
  re-scope admission to what the scheduler cannot already do (B4); batch
  queueing instead of `429` for `priority: batch` (B5); split multi-model
  serving into separate processes (**B6 — complete**, deletes `enforce_eager`
  coupling, the `max_model_len` 2048 clamp, and the multi-engine sequential-VRAM
  heuristics; DEC-059).
- **Phase C** (the differentiator): a signed run manifest and `X-Run-Id`
  (C1); `batch.co_batched_request_ids` (C2); `deterministic: true` wiring
  `VLLM_BATCH_INVARIANT=1`, refusing on unsupported hardware (C3); an
  oracle/conformance test suite (C4); a real Varex end-to-end run (C5);
  `make plan`/`make doctor` (C6).
- **Phase D** (consumer hardware, includes D5 — engine factory,
  optional-`vllm`, a `llama-server` proxy backend): requires its own
  accepted ADR authorizing a concrete second backend before it can begin;
  DEC-047 does not authorize it.

### B1 status: complete

`VLLMEngine` now constructs and drives vLLM's `AsyncLLM` (v1 async engine
client) directly; `LLM` + `EngineDriver` and the DEC-038/039/043 race-class
history are deleted. Full rationale, verification evidence, and the 8
implementation decisions: DEC-058, `openspec/changes/archive/` (change id
`migrate-async-llm-engine`). Summary:

- `_POOL_STEP_LOCK` deleted outright — each `AsyncLLM` instance owns an
  independent engine-core process, so the in-process race it guarded against
  has no equivalent under `AsyncLLM`.
- `generate()` is derived from `generate_stream()` — exactly one code path
  calls into `AsyncLLM.generate()`, preserving the DEC-050 single-source
  guarantee for terminal usage metadata.
- `pool_size > 1` remains an open question — B1 neither guarantees nor
  forbids it; B6 remains the only phase authorized to redesign multi-engine
  serving.
- Cancellation and per-request timeout are now real, engine-side signals
  (previously a disconnected or timed-out request's computation ran to
  completion regardless) — see `CHANGELOG.md`.
- Health (`AsyncLLM.errored`) and KV-cache introspection
  (`self._llm.vllm_config.cache_config`) are sourced directly from
  `AsyncLLM`; no repo-local dead-flag or driver wrapper remains.
- Zero edits landed in `api/`, `services/`, or `routing/` — the migration is
  contained entirely within `engines/` as scoped.

This directly affects the first deferred-assumption bullet above
(`_warm_ttft_ms`): `AsyncLLM` is now the live runtime, so that assumption is
active, not merely anticipated. It is not re-verified here — a benchmarking
change is needed to confirm or refute it, tracked as future Phase B/C work
(Phase A audit, "Advisor assumption review").

### B2 status: complete

`GET /metrics` mounts vLLM's own default `PrometheusStatLogger` output as a
Prometheus text-exposition endpoint, additive alongside the pre-existing
`GET /v1/metrics` JSON summary — no other InferenceX API surface changed. No
OpenSpec proposal existed for B2 before this change; it is tracked as
`openspec/changes/expose-native-engine-metrics/`. Summary, verified directly
against the shipped implementation rather than restated from `design.md`:

- **Registry identity was confirmed empirically, not assumed.** vLLM's
  default `PrometheusStatLogger` (`vllm.v1.metrics.loggers`) registers its
  `vllm:*` series directly onto the process-global `prometheus_client.
  REGISTRY` — confirmed by constructing a real `PrometheusStatLogger`
  against a real `VllmConfig` and observing new `vllm:`-prefixed collectors
  appear in `REGISTRY._names_to_collectors`. `api/main.py` mounts
  `prometheus_client.make_asgi_app()` with no registry argument and no
  `vllm` import in `api/` — the Engine Boundary (DEC-047) stays clean.
- **`GET /metrics` is excluded from `ObservabilityMiddleware`**, so scraping
  it never enters `InMemoryStorage` or affects `GET /v1/metrics`'s
  aggregates (`total_requests`, `avg_latency_ms`, etc.). This is the only
  path added to the exclusion. **`GET /health` is not excluded** and was
  never claimed to need to be: it was already recorded into `/v1/metrics`
  before this change and still is — pre-existing behavior this change does
  not touch.
- `pool_size > 1` now logs one startup `WARNING` naming the metric-label
  collision risk of multiple `AsyncLLM` instances' default stat loggers
  sharing one process-wide registry. Recorded, not guarded against — no new
  construction-time validation in either direction, consistent with B1
  Decision 3's stance that `pool_size > 1` is neither guaranteed nor
  forbidden.
- `prometheus-client` is now a direct `pyproject.toml` dependency (previously
  transitive via `vllm` only), because `api/main.py` imports its public API
  directly.
- No wire-format or existing-endpoint behavior changed: `/v1/chat/
  completions`, `/v1/models`, `/health`, and `/v1/metrics` are unaffected —
  the only addition is the new `GET /metrics` route.

### B3 status: complete

Per-request engine timing (`timing.queue_time_ms`, `.prefill_time_ms`,
`.decode_time_ms`, `.inference_time_ms`) is now additive on
`ChatCompletionResponse` and the streaming terminal event. Tracked as
`openspec/changes/archive/` (change id `expose-per-request-engine-timing`).
Summary:

- **Every field was verified empirically against vLLM 0.22.1 before
  design**, not assumed: `RequestOutput.metrics` is a
  `vllm.v1.metrics.stats.RequestStateStats`, confirmed populated by a real
  `AsyncLLM.generate()` call. Its four monotonic fields (`queued_ts`,
  `scheduled_ts`, `first_token_ts`, `last_token_ts`) were traced to their
  exact `time.monotonic()` call sites inside the engine-core process.
- **Engine timing and HTTP-boundary timing are clock-domain separated by
  construction**, not by convention. `arrival_time`
  (`RequestStateStats`) and `first_token_latency` are frontend
  wall-clock (`time.time()`) values and are deliberately never read —
  `first_token_latency` in particular would duplicate `GET /v1/metrics`'s
  existing TTFT (DEC-049).
- **The four exposed fields are vLLM's own formula**, lifted from its
  internal `do_tracing()` span builder — not independently derived. A
  fifth field ("scheduler delay" distinct from "queue wait") was
  considered and rejected: vLLM 0.22.1 tracks exactly one
  `QUEUED`→`SCHEDULED` interval per request, not two.
- Derivation extends `derive_terminal_metadata()` (the same DEC-050
  single-source call site `usage`/`finish_reason` already used) — no
  second read of engine state, so streaming and non-streaming report
  identical timing for the same request, structurally.
- Reads are defensive (`getattr` with a `None` default): any missing or
  non-numeric field on `RequestStateStats` — an internal vLLM type with no
  documented field-stability guarantee — degrades `timing` to `None`
  entirely, never a partially populated block.

## 11. ADR index

Phase A ADRs, in landing order (full text: `docs/DECISIONS.md`):

| ADR | Title | OpenSpec unit |
|---|---|---|
| DEC-047 | Engine Boundary and backend plurality | — (standalone) |
| DEC-048 | Lint rule set and type-check baseline for the CI gate | OS-1 |
| DEC-049 | Widen `BaseEngine.generate_stream` for truthful streaming usage | OS-2 |
| DEC-050 | Streamed token counts before OS-2 are superseded and incomparable | OS-2 |
| DEC-051 | Seed support / Deterministic Generation Contract | OS-3 |
| DEC-052 | `strict` may only convert substitution into rejection | OS-4 |
| DEC-053 | Pre-generation and post-generation metadata lifecycle | OS-4 |
| DEC-054 | Benchmark suite identity is a pinned content hash | OS-5 |
| DEC-055 | `suite_version` is a necessary but not sufficient comparability key | OS-5 |
| DEC-056 | Advisor score reflects measured quantities only | OS-6 |
| DEC-057 | The benchmark VRAM number is device occupancy (`vram_device_occupied_gib`) | OS-6 |

Conceptual dependencies among Phase A ADRs (not chronological landing order):

```text
DEC-047
   │
   ├── DEC-049
   │      │
   │      ├── DEC-050
   │      ├── DEC-052
   │      └── DEC-053
   │
   ├── DEC-051
   │
   ├── DEC-054
   │      │
   │      └── DEC-055
   │
   └── DEC-056
          │
          └── DEC-057
```

DEC-048 (CI/static-analysis gate) is Phase A hygiene under the same program
but sits outside this dependency tree: it does not depend on, and is not
depended on by, the Engine Boundary or later Phase A contracts.

Predecessor ADRs referenced above but outside Phase A: DEC-007 (vLLM required
dependency), DEC-011/DEC-021 (multi-model serving), DEC-023 (SSE streaming),
DEC-036/DEC-037 (VRAM sizing), DEC-038/DEC-039 (shared `EngineDriver`
thread). See `docs/DECISIONS.md` for the full record from DEC-000.

Supporting historical documents this reference replaces the need to consult
directly: `docs/REVIEW-2026-08-03-architecture.md` (origin roadmap and
Phase A–E naming), `docs/REVIEW-2026-08-04-os4-architecture.md` and
`docs/REVIEW-2026-08-04-os4-architecture-freeze.md` (OS-4 review),
`docs/REVIEW-2026-08-04-os5-os6-finalization.md` (OS-5/OS-6 review),
`docs/HANDOFF-2026-08-04-os4.md`, `docs/PHASE-A-EXECUTION-PLAN.md`, and the
OpenSpec changes themselves. OS-2/OS-3/OS-4 are archived under
`openspec/changes/archive/` (`2026-08-04-add-truthful-token-accounting`,
`2026-08-04-add-deterministic-generation`,
`2026-08-04-add-truthful-runtime-resolution`). OS-1
(`2026-08-04-add-ci-and-static-analysis-gate`) and OS-5/OS-6
(`verify-benchmark-suite-identity`, `add-honest-advisor-scoring`) are
implemented but sit directly under `openspec/changes/`, not yet archived.
