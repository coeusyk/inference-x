# OS-3 — Deterministic generation

- Change ID: `add-deterministic-generation`
- Milestone: Phase A, unit OS-3 (`docs/PHASE-A-EXECUTION-PLAN.md` §4)
- Traces to: review task A3
- Depends on: OS-1 (complete and enforced)
- Enables: OS-4 (request-side `seed` that OS-4 may echo in `resolved`)
- Parallel historically with: OS-2 (archived; no code dependency)

## Why

Clients that pin `seed` (notably Varex) today have that field silently dropped by
Pydantic because `ChatCompletionRequest` has no such field. That is worse than
unsupported: the client believes the run is pinned when it is not. Phase A OS-3
establishes the Deterministic Generation Contract at the sampling-input layer so
an explicit seed reaches the live backend sampler unchanged — without claiming
end-to-end determinism.

## What Changes

- Add optional `seed: Optional[int] = None` to `ChatCompletionRequest` (after
  `stream_options`).
- Thread `seed` into vLLM `SamplingParams` from `VLLMEngine._sampling_params`
  when `seed is not None`, forwarding the integer **unchanged** (including `-1`).
- When `seed is None`, omit the key — pre-OS-3 sampling kwargs preserved.
- Record DEC-051 (seed support / Deterministic Generation Contract).
- Tests: forwarding, schema acceptance, sequential identity, concurrency `xfail`
  citing Phase C3.
- **Not BREAKING** for clients that omit `seed`. **Intentional behavior change
  (B1)** for clients that already send `seed`: it now affects sampling.

## Capabilities

### New Capabilities

None. Deterministic generation is expressed as a requirement on the existing
platform capability, not a new OpenSpec capability folder.

### Modified Capabilities

- `platform`: optional request `seed` MUST be accepted and MUST reach the live
  sampler unchanged when set; documentation MUST NOT claim end-to-end
  determinism; response MUST NOT echo seed in this change.

## Impact

| Area | Effect |
|---|---|
| `src/inference_x/schemas/chat.py` | Append `seed` after `stream_options` |
| `src/inference_x/engines/vllm_engine.py` | `_sampling_params` only |
| `tests/unit/test_vllm_sampling.py` (+ identity/xfail) | New assertions |
| `docs/DECISIONS.md` | DEC-051 |
| Optional README / UNDERSTANDING | Doc language: “honoured”, not “reproducible” |
| `benchmarks/runner.py` | **Unchanged** (throughput-oriented defaults) |
| Engine Boundary / driver / admission / SSE | **Unchanged** |

---

## Summary

OS-3 establishes the **Deterministic Generation Contract** at the sampling-input
layer: the client may request a seed; the runtime accepts and forwards it
unchanged to the live backend sampler; the backend owns honouring it. This is
not end-to-end determinism, batch invariance, or replay. Implementation is the
thin vertical slice named in the Phase A plan (§4 OS-3).

## Motivation

1. `ChatCompletionRequest` has no `seed`. Extra JSON fields are ignored —
   Varex’s pinned seed is silently dropped (`docs/PHASE-A-EXECUTION-PLAN.md` §2.2).
2. `_sampling_params` builds `SamplingParams` from temperature / max_tokens /
   top_p / optional repetition_penalty only — no seed path.
3. Phase A acceptance criterion D.9 requires seed to reach `SamplingParams`
   without claiming end-to-end determinism (echo is OS-4; batch invariance is
   Phase C3).

## Scope

1. **Request field.** `seed: Optional[int] = None` on `ChatCompletionRequest`,
   appended after `stream_options`.
2. **Engine wiring.** In `_sampling_params`, when `request.seed is not None`,
   set `kwargs["seed"] = request.seed` (forward unchanged). When `None`, omit.
3. **Tests.** Forwarding present/absent; schema; sequential identity; concurrency
   `xfail` → Phase C3.
4. **DEC-051.** Record the Deterministic Generation Contract and ownership
   one-liner.

## Explicit non-goals

- **No** response echo / `resolved` / `warnings` / `strict` — OS-4.
- **No** `count_prompt_tokens` on `BaseEngine` — OS-4.
- **No** Engine Boundary change (`engines/base.py` untouched).
- **No** `EngineDriver` / AsyncLLM / Memory Manager / scheduler work — Phase B+.
- **No** `deterministic: true` / `VLLM_BATCH_INVARIANT` — Phase C3.
- **No** run manifests, co-batched IDs, oracle / `verify-determinism` — Phase C.
- **No** `suite_version` — OS-5; **no** advisor / VRAM rename — OS-6.
- **No** `engines/registry` factory / `api/deps.py` changes.
- **No** `inference_x/execution/`, second backend, optional-extra `vllm` — DEC-047.
- **No** growth of the DEC-048 mypy baseline.
- **No** pinning seeds in `benchmarks/runner.py` — deterministic benchmarking
  deferred.
- **No** claims of end-to-end determinism in documentation.

---

## Deterministic Generation Contract

Normative for OS-3. Worded so Phase C can **add** guarantees without rewriting
this contract.

### Guaranteed behavior (sampling-input contract)

**G1. Acceptance.** When the client supplies `seed` on
`POST /v1/chat/completions`, the platform accepts it as a first-class optional
field on `ChatCompletionRequest` (`Optional[int] = None`). It is not silently
dropped by the schema.

**G2. Unchanged forward.** When `seed is not None` and the live vLLM engine
builds `SamplingParams`, the runtime passes that integer into the backend
sampling API **exactly as received**. The runtime does not rewrite, clamp, or
normalize the value.

**G3. Omission equivalence.** When `seed is None` (absent or default), the
sampling kwargs **omit** `seed`, preserving pre-OS-3 sampling construction for
all other parameters.

**G4. Path parity.** Streaming and non-streaming generation use the same
`_sampling_params` builder, so seed honour does not diverge by transport.

**G5. Honesty.** Documentation and APIs state that seed is **honoured** (reaches
the sampler). They do **not** claim that the server is deterministic or that
runs are reproducible end-to-end.

### Intentionally non-guaranteed behavior

These are **not** OS-3 defects. Phase C may add opt-in contracts that narrow this
set without changing G1–G5.

**N1.** Output identity under concurrent / batched requests.  
**N2.** Output identity across hardware, drivers, CUDA graphs / JIT warmup,
prefix cache, or speculative decode.  
**N3.** Bit-identical outputs across backend versions.  
**N4.** Replay from recorded traces, run manifests, or co-batched request IDs.  
**N5.** That the runtime invents determinism when the client did not request a
seed (or when the backend cannot apply it on a degraded path).  
**N6.** That default or historical benchmark runs are deterministic.  
**N7.** Echo of the effective seed on the response — OS-4.  
**N8.** Backend-specific interpretation of sentinel values (e.g. `-1`) — owned
by the backend, not normalized by Inference-X.

### Future extension points

- Request schema may gain additional deterministic-generation fields in future
  milestones without changing the semantics of `seed`.
- Response metadata may expose effective seed in OS-4.
- Replay metadata, scheduler determinism, and batch invariance remain separate
  concerns and must not redefine this contract.

---

## Ownership

| Role | Owns | Must not |
|---|---|---|
| **Client** | Requesting determinism: supply `seed`, choose params, sequential vs concurrent load | Assume e2e determinism without Phase C contracts |
| **Backend** (vLLM today) | Honouring the seed in its sampling implementation; backend-native sentinel semantics | Be bypassed by a runtime that fabricates “deterministic” outputs |
| **Runtime** (schema + wiring) | Accept and forward `seed` unchanged on the live path; keep omit-seed ≡ pre-OS-3 | Invent determinism; normalize backend seed semantics; echo seed; claim reproducibility; pin benchmarks |

> The client owns requesting determinism; the backend owns honouring it; the
> runtime must not invent it.

### Field ownership (reaffirm OS-2 R4)

| Field | Owner | Location |
|---|---|---|
| `stream_options.include_usage` | OS-2 (done) | `ChatCompletionRequest` |
| `seed` | **OS-3** | `ChatCompletionRequest` |
| `strict` | OS-4 | `ChatCompletionRequest` |
| `warnings` / `resolved` (incl. seed echo) | OS-4 | `ChatCompletionResponse` |

---

## Resolved ambiguities

Implementation requires no architectural interpretation.

### R1 — Omit vs `seed=None` in kwargs

**Omit the key** when `request.seed is None`. Do not pass `seed=None` into
`SamplingParams`. Matches the existing sparse-kwargs style for optional fields.

### R2 — `seed == -1`

Accept any `int` (no `ge=0`). When `seed is not None` (including `-1`), forward
**unchanged**. Inference-X does **not** normalize backend semantics: it does not
map `-1` → `None`, reject `-1`, or emit a warning. Interpretation of sentinels is
solely the backend’s. No HTTP 422 for negative seeds in OS-3.

### R3 — Degraded / no-vLLM path

When `_VLLM_AVAILABLE == False`, no `SamplingParams` is built; seed has no
effect. Do **not** invent a warning or HTTP error for that (OS-4 owns
degradation signalling). Preserve the single content-message degraded stream
behaviour (no terminal `finish_reason="error"`).

### R4 — Documentation language

Say seed is **honoured** / **reaches the sampler**. Never “runs are reproducible”
or “the server is deterministic” (Phase A Rank 6 / AC D.9).

### R5 — Benchmarks

`benchmarks/runner.py` is **unchanged**. Defaults remain throughput-oriented and
unseeded. Deterministic benchmarking is intentionally deferred. Harnesses that
need reproducibility (e.g. Varex) send `seed` themselves.

### R6 — Identity vs concurrency tests

Identity requires **same prompt**, **same parameters** (including
`temperature=0`), **same seed**, and **sequential execution** (one request fully
completed before the next; single in-flight). Multi-in-flight with seed is
`xfail` citing Phase C3.

### R7 — Dependency on OS-4

Wording is **Enables OS-4**, not “Blocks OS-4 (soft)”. The Phase A edge is thin:
OS-4 can ship without seed echo and add it later. OS-3 provides the request-side
field OS-4’s `resolved` echo consumes.

---

## Compatibility invariants

Mandatory implementation review checks. Every one must remain true. Any
violation is a defect, not a trade-off.

**C1.** Omitting `seed` (or `seed: null`) yields the same `SamplingParams` kwargs
as pre-OS-3 for identical other request fields (temperature, max_tokens, top_p,
repetition_penalty rules unchanged).

**C2.** `ChatCompletionResponse` gains no field and loses no field. No seed echo,
no `resolved`, no `warnings`, no `strict`.

**C3.** OS-2 streaming contract is unchanged: content → terminal `finish_reason`
→ optional usage → `[DONE]`; timeout path still error + `[DONE]`.

**C4.** `BaseEngine` / Engine Boundary is untouched. No new abstract method; no
`count_prompt_tokens`.

**C5.** `EngineDriver` dead-flag, locking, submission-after-death (DEC-043), and
restart policy are unchanged.

**C6.** Admission control is unchanged. `routing/admission.py` is not modified.

**C7.** `EnginePool`, `api/deps.py`, and the composition root are unchanged. No
route handler imports a concrete engine for seed.

**C8.** `benchmarks/runner.py` is unchanged. Default benchmark requests remain
unseeded and throughput-oriented.

**C9.** Observability middleware and `/v1/metrics` schema/behaviour from OS-2 are
unchanged (no seed-derived metrics).

**C10.** Degraded `_VLLM_AVAILABLE == False` stream path remains a single content
message with no terminal `finish_reason="error"` and no new seed-related error or
warning.

**C11.** `vllm` remains a required dependency. No `inference_x/execution/`. No
second backend. `ModelEntry.engine` remains `Literal["vllm"]`.

**C12.** The DEC-048 mypy baseline does not grow.

**C13.** `docs/PHASES.md` is untouched.

**C14.** Playground clients are not required to change; additive optional `seed`
is ignored if unused.

**C15.** When `seed` is set, it is **additive only**: other sampling kwargs
continue to be derived exactly as today.

## Intentional behavioral changes

Exhaustive. Nothing outside this list may change.

**B1.** A request that includes `seed` causes that value to be passed into live
vLLM `SamplingParams`. Pre-OS-3, the same JSON field was silently dropped and
did not affect sampling.

---

## Validation strategy

1. **Forwarding unit tests** (`tests/unit/test_vllm_sampling.py`): `seed` set →
   kwargs contain the exact int (including `-1`); `seed` absent → key omitted.
2. **Schema / route validation**: request with `seed` accepts; wrong types still
   422.
3. **Identity**: two completions with same prompt, same parameters (including
   `temperature=0`), same seed, sequential execution → identical assistant
   content. Follow existing GPU/live skip patterns if needed.
4. **Concurrency**: multi-in-flight with seed → `xfail`, reason cites Phase C3.
5. **Gates:** full unit suite, `ruff check .`, `mypy src/`, DEC-048 baseline
   unchanged, `openspec validate --strict`.
6. **Review checklist:** every C1–C15 holds; only B1 changed.

## Rollback strategy

Revert the schema field, `_sampling_params` wiring, tests, and DEC-051. No metric
supersession (unlike OS-2). Clients that began sending `seed` return to the
pre-OS-3 silent-drop failure mode — document that in the revert note if needed.

## Acceptance criteria

1. `ChatCompletionRequest.seed: Optional[int] = None` accepted on the wire.
2. When set, live `_sampling_params` passes that exact int into
   `SamplingParams`; when unset, kwargs omit `seed` (G1–G3).
3. Stream and non-stream both use `_sampling_params` (G4).
4. Forwarding and schema unit tests pass; identity test as specified; concurrency
   `xfail` cites Phase C3.
5. No response field for seed; docs do not claim e2e determinism (G5, N7).
6. DEC-051 accepted with Contract + ownership one-liner.
7. C1–C15 hold; only B1 changes; DEC-048 baseline unchanged.
8. `ruff`, `mypy`, full unit suite, and `openspec validate --strict` pass.

## Repository impact

| Path | Action |
|---|---|
| `src/inference_x/schemas/chat.py` | Append `seed` |
| `src/inference_x/engines/vllm_engine.py` | `_sampling_params` only |
| `tests/unit/test_vllm_sampling.py` | Forwarding tests |
| `tests/unit/` (identity/xfail module as needed) | Identity + concurrency |
| `docs/DECISIONS.md` | DEC-051 |
| Optional `README.md` / `docs/UNDERSTANDING-INFERENCE-X.md` | Honour language |

## Do not touch

`engines/base.py`, `engines/driver.py`, `services/chat_service.py` (SSE),
`routing/admission.py`, `benchmarks/runner.py`, `observability/middleware.py`,
`api/deps.py`, `engines/registry`, OS-4 response fields, Phase B/C surfaces.
