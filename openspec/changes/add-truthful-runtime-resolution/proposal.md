# OS-4 — Truthful runtime resolution and typed degradation

## Executive summary

Make every substitution the server performs visible to the client that asked for
something else.

Today `AdmissionController` may clamp `max_tokens`, may skip a gate entirely when
the number it needs is unavailable, and may estimate a prompt-token count from
character length — and the client learns none of it. The response is silent about
every one of those decisions. This change gives the substitution a name
(**Effective Request**), a wire surface (`resolved`), a typed degradation channel
(`warnings`), and an opt-in rejection mode (`strict`).

It also closes the DEC-047 §3 capability-hygiene item that lives on the same code
path: `count_prompt_tokens` becomes a declared method on `BaseEngine` instead of a
`getattr` probe.

Two invariants govern the shape and are recorded as ADRs: **DEC-052** (already
accepted) fixes what `strict` may and may not do; **DEC-053** (written by this
change) fixes which metadata may be emitted before generation.

Admission still decides exactly what it decided before. This change makes
degradation *observable*, never fail-closed (DEC-047 §4).

## Motivation

`docs/PHASE-A-EXECUTION-PLAN.md` §9 C.6 requires that *a client can determine, from
the response alone, every parameter the server substituted for one the client
supplied.* Three concrete gaps stand between the repository and that criterion:

1. **Silent clamping.** `chat_service.complete()` and `stream_response()` both build
   `request.model_copy(update={"max_tokens": admitted.effective_max_tokens})` and
   then discard the fact that a substitution happened. A client that asked for 4096
   output tokens and received 512 has no way to distinguish that from a model that
   simply stopped early.

2. **Silent degradation.** All three admission gates fail open when their input is
   unavailable — no resolved tier, no reported KV capacity, no tokenizer. That
   posture is correct and DEC-047 §4 requires preserving it. But a gate that
   silently does not run is indistinguishable from a gate that ran and passed, both
   to the client and to whoever later has to decide whether tightening is safe.

3. **Undeclared capability.** `routing/admission.py:91` discovers
   `count_prompt_tokens` through `getattr` with a silent chars/4 fallback. DEC-047
   §3 names this capability **durable** and permits declaring it. Because OS-4 is
   already editing this exact discovery path to make its degradation observable,
   declaring it here costs one method and prevents a second edit to the same lines
   later.

## Scope

1. **`ResponseWarning`** in `schemas/chat.py` — `{type, code, message, field}`, with
   `type` drawn from a closed two-member set: `substituted` (the server ran
   something different from what the client asked for) and `degraded` (the server
   could not verify something and proceeded anyway).

2. **`ResolvedRequest`** in `schemas/chat.py` — the serialized Effective Request.
   Its contents follow a derivability rule rather than curation: a field appears in
   `ResolvedRequest` **iff** it exists on `ChatCompletionRequest`, minus `messages`
   (content, not a parameter) and minus transport/policy controls (`stream`,
   `stream_options`, `strict`). What remains: `model`, `temperature`, `max_tokens`,
   `top_p`, `max_context_tokens`, `max_output_tokens`, `priority`, `seed`.

3. **`strict: bool = False`** on `ChatCompletionRequest`, appended last (OS-2 R4
   ownership table).

4. **`resolved: ResolvedRequest | None` and `warnings: list[ResponseWarning]`** on
   `ChatCompletionResponse`. Populated by `ChatService`, never by an engine — the
   Engine Boundary does not learn about admission.

5. **`AdmissionResult.warnings: tuple[ResponseWarning, ...]`** — admission produces
   the warnings because admission is where the substitutions and the skips happen.

6. **`StrictModeViolationError(ValueError)`** in `routing/admission.py`, mapped to
   400 by the existing sanitized `value_error_handler` with no new registration —
   the same pattern `ContextTooLongError` already uses.

7. **One pre-generation SSE event** emitted by `ChatService.stream_response()` after
   `admit()` returns and before the first content event, carrying `resolved` and
   `warnings` with an empty `choices` array.

8. **`BaseEngine.count_prompt_tokens(request) -> int | None`** — declared,
   non-abstract, default `return None`. Admission calls the declared method instead
   of probing with `getattr`.

9. **`observability/middleware.py` TTFT anchored to the first content event** rather
   than the first raw chunk, so `/v1/metrics` TTFT keeps the meaning it has today.

10. **DEC-053** in `docs/DECISIONS.md`, status `accepted`: the pre-generation and
    post-generation metadata lifecycle.

### Warning codes

The closed set this change introduces. Adding a code later is a spec change.

| `type` | `code` | `field` | Emitted when |
|---|---|---|---|
| `substituted` | `max_tokens_clamped_to_context` | `max_tokens` | Output clamped to fit the context ceiling |
| `substituted` | `max_tokens_clamped_to_kv_budget` | `max_tokens` | Output clamped to fit remaining KV budget |
| `degraded` | `prompt_tokens_estimated` | `messages` | No engine tokenizer; chars/4 heuristic used as the gate input |
| `degraded` | `kv_gate_skipped` | `null` | Engine reports no `kv_capacity_tokens`; KV gate did not run |
| `degraded` | `sequence_gate_skipped` | `null` | No resolved VRAM tier; sequence-concurrency gate did not run |

**`strict` rejects on `substituted` only.** A `degraded` warning under `strict`
still admits the request. This is not a softening — it is DEC-047 §4 enforced:
rejecting on `degraded` would convert fail-open into fail-closed, which this
milestone forbids.

## Non-goals

- **No `kv_capacity_tokens` on `BaseEngine`.** DEC-047 §3 marks it *provisional* and
  forbids freezing it as a cross-backend contract; Phase B4 may delete it. Both
  `getattr` sites (`routing/admission.py:212`, `api/routes/metrics.py:43`) stay
  exactly as they are.
- **No admission-decision change.** Every request admitted today is admitted after
  this change with the same `effective_max_tokens`, unless the client opted into
  `strict`. Nothing becomes fail-closed.
- **No per-request timings** (Phase B3). The pre-generation event carries no timing
  field, and `ChatStreamChunk` is not widened. The middleware change is a
  *preservation* of today's TTFT meaning against this change's own blast radius, not
  new timing functionality.
- **No AsyncLLM migration, no `EngineDriver` change** (Phase B1).
- **No `inference_x/execution/`, no backend-neutral DTOs, no second backend, no
  optional-extra `vllm`** — DEC-047 non-decisions.
- **No `engines/registry.py` factory and no `api/deps.py` change.** Deferred to the
  Phase A/B seam (plan §2.4).
- **No scheduler, batching, or pool-config change.** `utils/vllm_pool_config.py` is
  not touched; tier knobs are process-level state and do not enter `resolved`.
- **No `playground/` change.** It is a client and stays unmodified. Its two
  `len(content.split())` displays remain a recorded follow-up.
- **No change to `VLLMEngine.count_prompt_tokens`'s internal chars/4 fallback.** The
  engine's own tokenizer-failure path keeps self-estimating and logging at debug.
  See Known limitation below.
- **No `n > 1` support, no per-choice warnings.** `warnings` and `resolved` are
  per-request.
- **No growth of the DEC-048 mypy baseline** (`vllm_engine`, `driver`,
  `chat_service`, `api/deps`).

### Known limitation, stated rather than fixed

When `VLLMEngine`'s tokenizer raises, the engine returns its own chars/4 estimate
rather than `None`, so admission cannot distinguish it from a real count and emits
no `prompt_tokens_estimated` warning. That path is logged by the engine at debug
level today and remains so. Changing it means editing the engine's fallback
contract, which is outside this change's `Contains` list (plan §4). Recorded here
as a residual gap against §9 C.8 rather than silently absorbed into scope.

## Repository impact

### Modified — source

| File | Change |
|---|---|
| `src/inference_x/schemas/chat.py` | Add `ResponseWarning`, `ResolvedRequest`; add `strict` to `ChatCompletionRequest`; add `resolved` + `warnings` to `ChatCompletionResponse` |
| `src/inference_x/engines/base.py` | Declare `count_prompt_tokens`, non-abstract, default `None` |
| `src/inference_x/routing/admission.py` | Call the declared method; build `warnings`; add `StrictModeViolationError`; widen `AdmissionResult`; update module docstring |
| `src/inference_x/services/chat_service.py` | Attach `resolved`/`warnings` to the non-streaming response; emit the pre-generation event on the streaming path |
| `src/inference_x/observability/middleware.py` | Anchor TTFT to the first content event |
| `src/inference_x/engines/vllm_engine.py` | Docstring only — the "BaseEngine doesn't declare this" comment becomes false |

### Modified — tests

`tests/unit/test_schemas.py`, `tests/unit/test_admission.py`,
`tests/unit/test_chat_service.py`, `tests/unit/test_routes.py`,
`tests/unit/test_observability.py`.

### Modified — docs

`docs/DECISIONS.md` (DEC-053).

### Verified unchanged

`src/inference_x/benchmarks/runner.py` — already anchors TTFT on non-empty content
(`runner.py:77`) and already tolerates `choices: []` via
`(chunk.get("choices") or [{}])[0]` (`runner.py:75`). No edit required; asserted as
C10.

`playground/streaming.py`, `src/inference_x/api/routes/metrics.py`,
`src/inference_x/engines/driver.py`, `src/inference_x/engines/registry.py`,
`src/inference_x/api/deps.py`, `src/inference_x/utils/vllm_pool_config.py`.

## Dependencies

- **OS-1** (archived) — CI and type gate must be enforcing before behavioural work.
- **OS-2** (archived) — this change extends the SSE generator and the streaming
  requirement OS-2 established. Its `ChatStreamChunk` and terminal/usage events are
  preconditions.
- **OS-3** (archived) — `seed` must exist on the request before `resolved` can echo
  it. DEC-051 deferred the echo to this change.
- **DEC-052** — accepted in the tree; this change implements it.
- **Blocks:** nothing.

## Risks

**R1 — The pre-generation event breaks a stream consumer.** *Likelihood: low.*
All three first-party consumers were re-verified against this change's event shape,
not inferred: `middleware._extract_sse_usage` returns `None` for any payload without
a dict `usage`; `benchmarks/runner.py:75` resolves `choices: []` to `{}` and skips
an empty `content`; `playground/streaming.py` yields nothing for a line producing no
token. C4–C6 assert each. Third-party consumers are covered by the OpenAI-shaped
`choices: []` precedent the usage event already set.

**R2 — TTFT silently changes meaning.** *Likelihood: medium if unguarded, and the
reason middleware is in scope.* `middleware.py` currently timestamps the first raw
chunk. Left alone, every `/v1/metrics` TTFT would silently gain the admission
latency and become incomparable with figures recorded before this change — a
DEC-050-class discontinuity introduced by accident. Task 5 and C11 close it.

**R3 — `strict` drifts from the warning set.** *Likelihood: medium over time.*
Two predicates encoding one policy will diverge. Mitigated structurally: `strict`
raises inside the same branch that appends the `substituted` warning, and B4 is
verified by a test that enumerates both sets rather than sampling them (DEC-052 §2).

**R4 — Someone tightens admission while making it observable.** *Likelihood: medium
— the code reads as if it wants tightening.* DEC-047 §4 forbids it. Guarded by C7,
C8 and by `strict` rejecting on `substituted` only.

**R5 — `resolved` becomes a junk drawer.** *Likelihood: low now, high over three
phases.* Mitigated by the derivability rule being normative rather than advisory
(spec requirement, not a docstring), and by DEC-053 recording that the prologue is a
superset of `resolved` so future prologue payloads do not widen `resolved`.

**R6 — `chat_service.py` conflict resolution loses a change.** *Likelihood: low.*
OS-2 is archived and merged; the conflict surface named in plan §10 Rank 5 no longer
exists. The OS-2 event-ordering tests remain in `test_chat_service.py` and must still
pass unmodified except for the added prologue assertion.

## Compatibility invariants

Each must hold when this change is archived.

- **C1** — `ChatCompletionRequest` gains exactly one field (`strict`), appended last,
  defaulting to `false`. Every request valid before is valid after with identical
  behaviour.
- **C2** — `ChatCompletionResponse` gains exactly two fields (`resolved`,
  `warnings`), both with defaults. Existing clients that read `choices` and `usage`
  are unaffected.
- **C3** — No event is emitted after the usage event, on any path.
- **C4** — `observability/middleware._extract_sse_usage` returns `None` for the
  pre-generation event; no token figure is derived from it.
- **C5** — `benchmarks/runner.py` parses a stream containing the pre-generation event
  without modification, and its `tokens_generated` and `ttft_ms` are unchanged for an
  identical stream body.
- **C6** — `playground/streaming.py` parses a stream containing the pre-generation
  event without modification and yields no spurious token.
- **C7** — For every request, `admitted.effective_max_tokens` and
  `admitted.reserved_tokens` are identical before and after this change, given
  `strict` absent or false.
- **C8** — Every admission gate that failed open before still fails open. No
  `degraded` condition rejects, under either `strict` value.
- **C9** — `kv_capacity_tokens` is not declared on `BaseEngine`. Both `getattr` sites
  are byte-identical to their current form.
- **C10** — `benchmarks/runner.py`, `playground/`, `api/routes/metrics.py`,
  `engines/driver.py`, `engines/registry.py`, `api/deps.py` and
  `utils/vllm_pool_config.py` are unmodified.
- **C11** — `/v1/metrics` TTFT for an identical generation is unchanged by this
  change; the pre-generation event does not enter the measurement.
- **C12** — `inference_x/execution/` does not exist. No backend-neutral DTO package
  exists. `ModelEntry.engine` remains `Literal["vllm"]`.
- **C13** — No handler under `api/routes/` imports a concrete engine.
- **C14** — The DEC-048 mypy baseline has not grown.
- **C15** — `ChatStreamChunk` carries exactly its current three fields. The Engine
  Boundary's streaming element type is not widened by this change.
- **C16** — No engine constructs `resolved` or `warnings`. Both are attached by
  `ChatService`.

## Intentional behavioral changes

Only these. Anything else is a defect.

- **B1** — Every `/v1/chat/completions` non-streamed response now carries `resolved`
  and `warnings`.
- **B2** — Every streamed response now begins with one pre-generation event carrying
  `resolved` and `warnings`, before the first content event. This is emitted
  unconditionally, not gated on a request flag.
- **B3** — A clamped `max_tokens` now produces a `substituted` warning where it
  previously produced nothing.
- **B4** — `strict: true` now returns 400 where the default clamps. The rejecting
  condition set is exactly the `substituted` warning set.
- **B5** — A skipped gate or an estimated prompt count now produces a `degraded`
  warning and a structured log record where it previously produced at most a debug
  log.
- **B6** — `BaseEngine` declares `count_prompt_tokens`; admission calls it directly
  instead of probing with `getattr`.
- **B7** — `/v1/metrics` TTFT is measured from the first content event rather than
  the first SSE chunk. For streams before this change the two are identical, so
  recorded figures remain comparable — this preserves continuity rather than breaking
  it.

## Validation

1. **Streaming lifecycle test — authoritative.** Drive `ChatService.stream_response`
   and assert the **complete ordered event sequence** for four cases: default,
   `include_usage: true`, timeout, and a request that produces warnings. Assert order
   and exact event count. This test is the protocol in executable form and must not
   be weakened to accommodate an easier implementation. It extends, and must not
   replace, the OS-2 ordering tests already in `tests/unit/test_chat_service.py`.
2. **Pre-generation determinacy test.** Assert the pre-generation event is emitted
   before any content event, and that it is still emitted on the timeout path.
3. **Strict/default equivalence (DEC-052 §3).** For a request accepted under both
   modes with an identical seed, assert the completion content is byte-identical.
4. **Predicate enumeration (DEC-052 §2).** Assert the set of conditions raising
   `StrictModeViolationError` equals the set emitting a `substituted` warning —
   enumerated, not sampled.
5. **Fail-open preservation.** For each `degraded` condition, assert the request is
   admitted under both `strict` values and that `effective_max_tokens` matches the
   pre-change value.
6. **Consumer compatibility.** Assert C4, C5 and C6 by feeding each consumer a stream
   containing the pre-generation event.
7. **TTFT preservation.** Assert `/v1/metrics` TTFT is unaffected by the
   pre-generation event.
8. **Derivability.** A test that fails if `ResolvedRequest`'s field set diverges from
   `ChatCompletionRequest`'s minus the documented exclusions — so the rule is enforced
   by CI, not by review.
9. **Empty case.** An unclamped, capability-complete request returns `warnings: []` —
   asserted as empty, not absent and not null.
10. **Full suite, lint, types.** `uv run pytest tests/unit -q`, `uv run ruff check .`,
    `uv run mypy src/`.
11. **Invariant sweep.** `git diff --stat` touches no file outside Repository impact.
12. **`openspec validate --strict` passes for this change.**

## Rollback

Every element is additive at the wire level and independently revertible.

- Reverting the commit removes two response fields, one request field and one SSE
  event. No client that worked before this change depends on any of them.
- `resolved`/`warnings` carry no state and are recomputed per request; nothing
  persists that a rollback would strand.
- The `count_prompt_tokens` declaration reverts to `getattr` discovery with no
  behavioural difference, because the default returns `None` and the chars/4 fallback
  stays in admission either way.
- No stored metric changes meaning, so unlike OS-2 no supersession note is required
  (B7 preserves TTFT continuity rather than breaking it).
- DEC-052 is independent of this change and stays accepted. DEC-053 would be marked
  superseded rather than deleted.
