# OS-2 — Truthful token accounting

- Change ID: `2026-08-04-add-truthful-token-accounting`
- Milestone: Phase A, unit OS-2 (`docs/PHASE-A-EXECUTION-PLAN.md` §4)
- Traces to: review task A1
- Depends on: OS-1 (complete and enforced); DEC-049 accepted
- Blocks: OS-4, OS-6

## Summary

Make every completion-token count the server reports originate from the engine
rather than from counting whitespace-delimited words.

This requires widening the Engine Boundary: `BaseEngine.generate_stream` yields
`str`, and a string cannot carry `usage` or `finish_reason`. A streaming chunk
model is added to `schemas/chat.py`, `generate_stream` is widened to yield it, the
`EngineDriver` stream channel is widened to carry terminal metadata, and both
whitespace-approximation sites in `src/` are deleted.

This is the single Phase A change to the Engine Boundary, authorized by DEC-049
under DEC-047. It is not the start of a backend-neutral execution contract.

## Motivation

Three facts, all verified against the tree:

1. `observability/middleware.py::_count_sse_delta_tokens` computes
   `len(content.split())` over each SSE delta. That number feeds `/v1/metrics`.
2. `benchmarks/runner.py` does the same in its stream loop. That number feeds
   `tokens_per_sec`, which the advisor weights when recommending models.
3. `services/chat_service.py` emits SSE chunks carrying neither `finish_reason`
   nor `usage`, and ends the stream with a bare `data: [DONE]`.

Non-streaming `generate()` already returns engine-accounted
`ChatCompletionUsage`. Only the streaming path lies, and it lies by a
model-dependent margin because tokens are not words.

DEC-023 accepted this approximation explicitly, on the grounds that streamed
responses carried no usage event. DEC-049 removes that constraint. Phase A is the
milestone that makes reported numbers true; every later unit — the `resolved`
block in OS-4, the advisor rescoring in OS-6, the manifest in Phase C — is built
on top of this measurement. Correcting it later means every intervening artifact
recorded the wrong number precisely.

## Scope

1. **Streaming chunk model.** Add `ChatStreamChunk` to `schemas/chat.py` with
   exactly three fields:
   - `content: str` — delta text; empty string on a terminal-only chunk.
   - `finish_reason: Literal["stop", "length", "error"] | None` — `None` on
     content chunks.
   - `usage: ChatCompletionUsage | None` — `None` on content chunks; populated on
     the terminal chunk when the engine can account it.

   No other fields. Per-request timings are Phase B3 and must not be added here.

2. **Widen the engine contract.** `BaseEngine.generate_stream` returns
   `AsyncGenerator[ChatStreamChunk, None]`. Update `BaseEngine`'s docstring, whose
   present claim that "adding a second engine in a future phase must not require
   changes here" is now known to be false — DEC-047's problem statement §1
   anticipated exactly this.

3. **Widen the driver stream channel.** `EngineDriver.submit_stream`'s queue
   carries terminal metadata instead of discarding it. Today the queue element
   type is `str | BaseException | None` and `_dispatch` drops the `RequestOutput`
   fields on finish. The vLLM path cannot produce real `usage`/`finish_reason`
   without this.

4. **vLLM implementation.** `VLLMEngine.generate_stream` yields `ChatStreamChunk`,
   deriving `finish_reason` and `usage` from the terminal `RequestOutput` using
   the same accounting `generate()` already uses, so streamed and non-streamed
   counts agree for the same prompt and parameters.

5. **Request field.** Add `stream_options` to `ChatCompletionRequest`, an optional
   object with a single `include_usage: bool` field. See Resolved Ambiguities for
   the default and the exact wire behaviour.

6. **SSE serialization.** `ChatService.stream_response` maps `ChatStreamChunk`
   events onto the OpenAI streaming convention. Exact ordering is pinned in
   Resolved Ambiguities and is normative.

7. **Observability.** Delete `_count_sse_delta_tokens`. Repoint
   `_wrap_and_record_sse` to read `usage` from the terminal SSE chunk when
   present, and to record no completion-token figure when absent.

8. **Benchmark runner.** The stream loop reads `usage.completion_tokens` instead
   of counting words, and sets `stream_options.include_usage` on its own requests
   so the figure is available.

9. **Decision records.** Accept DEC-049 (currently `proposed`). Add one further
   ADR recording that all previously reported streamed token counts, and every
   throughput figure derived from them, are superseded and not comparable with
   figures produced after this change. Claim the next free DEC number at PR time.

## Explicit non-goals

- **No AsyncLLM migration and no `EngineDriver` deletion** (Phase B1). This change
  widens the driver's stream channel; it does not restructure the driver, alter
  its locking, its dead-flag semantics (DEC-043), or its restart policy.
- **No Prometheus passthrough** (Phase B2). No new metrics backend or exporter.
- **No per-request timings in the response body** (Phase B3). `ChatStreamChunk`
  carries no timing fields.
- **No admission changes** (Phase B4/OS-4). `routing/admission.py` is not touched.
- **No `seed`** — OS-3.
- **No `warnings`, `resolved`, `strict`, or `count_prompt_tokens` on
  `BaseEngine`** — OS-4.
- **No computed `suite_version`** — OS-5.
- **No advisor rescoring or VRAM field rename** — OS-6.
- **No archiving of settled changes or test-count corrections** — HK-1.
- **No `engines/registry` factory.** Deferred to the Phase A/B seam
  (`docs/PHASE-A-EXECUTION-PLAN.md` §2.4). `api/deps.py` is not touched.
- **No `inference_x/execution/`, no backend-neutral DTOs, no second backend, no
  optional-extra `vllm`** — DEC-047 non-decisions.
- **No playground token-count correction.** `playground/app.py` computes
  `len(content.split())` for its own display in two places. The playground is a
  *client*; its display is not a figure the server reports, and correcting it
  requires consuming the new usage chunk in the TUI. Recorded as a follow-up.
- **No `created`/`model` fields added to SSE chunks.** The current chunk shape is
  preserved apart from the additions pinned below.
- **No growth of the DEC-048 mypy baseline.** See Acceptance Criteria.

## Resolved ambiguities

These are decided here. Implementation requires no architectural interpretation.

### R1 — Chunk model name

**`ChatStreamChunk`**, defined in `src/inference_x/schemas/chat.py`. DEC-049
offered this name provisionally; it is now fixed. It sits beside the existing
wire models in the package DEC-047 describes as the boundary's current
vocabulary. No new module, no new package.

### R2 — `include_usage` default

**`stream_options.include_usage` defaults to `false`**, matching OpenAI. When
`stream_options` is absent entirely, behaviour is identical to `include_usage:
false`.

The reasoning is forced by the architecture, not chosen for taste.
`ObservabilityMiddleware` wraps the *response byte iterator* and parses `data:`
lines; it has no access to engine-internal objects. So `/v1/metrics` can only see
usage that is actually on the wire. Three options existed:

| Option | Verdict |
|---|---|
| Always emit the usage chunk, ignoring `include_usage` | Rejected. Makes the field a no-op and alters response shape for clients that did not ask. |
| Default `include_usage` to `true` | Rejected. Gratuitous divergence from the OpenAI contract this server claims to implement. |
| **Default `false`; emit the usage chunk only when requested** | **Accepted.** |

The consequence is deliberate and must not be "fixed" during implementation:
**when `include_usage` is false, `/v1/metrics` records no completion-token figure
for that request** — not an estimate. `metrics_service` already filters
`tokens_per_sec is not None`, so `avg_tokens_per_sec` degrades to `None` rather
than breaking. Replacing a wrong number with no number is the Phase A thesis
applied to this server's own telemetry.

The benchmark runner is a first-party client and sets `include_usage: true`
explicitly (Scope item 8), so benchmark figures remain populated and become true.

### R3 — Precise streaming ordering

Normative. `ChatService.stream_response` emits, in this exact order:

1. **Zero or more content events**, one per engine chunk with non-empty
   `content`:
   ```
   data: {"id":"<completion_id>","object":"chat.completion.chunk",
          "choices":[{"index":0,"delta":{"content":"<text>"},"finish_reason":null}]}
   ```
   `finish_reason` is present and `null`. This is the only change to the content
   chunk shape.

2. **Exactly one terminal choice event**, emitted when the engine yields a chunk
   with non-null `finish_reason`:
   ```
   data: {"id":"<completion_id>","object":"chat.completion.chunk",
          "choices":[{"index":0,"delta":{},"finish_reason":"<reason>"}]}
   ```
   A separate event with an empty delta — the last *content* event is not
   retroactively mutated, because the service cannot know a content event is the
   last one until the engine says so.

3. **One usage event, if and only if `include_usage` is true and the terminal
   engine chunk carried `usage`:**
   ```
   data: {"id":"<completion_id>","object":"chat.completion.chunk",
          "choices":[],"usage":{"prompt_tokens":N,"completion_tokens":N,"total_tokens":N}}
   ```
   `choices` is an empty array, per OpenAI.

4. **`data: [DONE]`**, always, exactly once, last.

**Error path is unchanged.** On stream timeout the service emits its existing
`data: {"error":"..."}` event, stops, and still emits `data: [DONE]`. No terminal
choice event and no usage event are emitted on that path. Emitting a
`finish_reason: "error"` event is *not* part of this change.

### R4 — Request-schema ownership across OS-2/OS-3/OS-4

Binding for all three units. A unit that adds a field outside its row is out of
scope.

| Field | Owner | Location |
|---|---|---|
| `stream_options.include_usage` | **OS-2** | `ChatCompletionRequest` |
| `ChatStreamChunk` | **OS-2** | `schemas/chat.py`, engine-internal |
| `seed` | OS-3 | `ChatCompletionRequest` |
| `strict` | OS-4 | `ChatCompletionRequest` |
| `warnings` | OS-4 | `ChatCompletionResponse` |
| `resolved` (incl. effective seed echo) | OS-4 | `ChatCompletionResponse` |

**OS-2 makes no change to `ChatCompletionResponse`.** The non-streaming response
already carries engine-accounted usage and is not modified.

### R6 — Driver stream-channel element type

The `submit_stream` queue element type becomes
**`ChatStreamChunk | BaseException | None`**, with `None` retained unchanged as
the end-of-stream sentinel.

Three designs were possible; this one is chosen because it is the least invasive
under C7:

| Design | Verdict |
|---|---|
| Replace the `None` sentinel with a terminal object | Rejected. Changes the end-of-stream protocol every consumer branches on, including `_fail_one`'s sibling paths. |
| Keep `None`, add a separate preceding terminal object of a new driver-local type | Rejected. Introduces a second chunk type for the same information — the duplication DEC-047 exists to prevent. |
| **Put `ChatStreamChunk` on the queue; keep `None` as the sentinel** | **Accepted.** |

Consequences that make this the safe choice: the sentinel path and the
`BaseException` path stay byte-identical, so `_fail_one` and the dead-driver
broadcast are untouched (C7); no new type is introduced anywhere; and
`VLLMEngine.generate_stream` becomes a pass-through rather than a translator,
which removes a place where streamed and non-streamed accounting could drift
apart.

`_dispatch` therefore puts a content-only `ChatStreamChunk` for each text delta,
and on `output.finished` puts a terminal `ChatStreamChunk` carrying
`finish_reason` and `usage` **before** putting the existing `None` sentinel.

### R5 — `article-final.md` handling

**Removed from this change's repository scope.**

`article-final.md` is gitignored under an explicit rationale comment (`# Private
writing — not for the public repo`) — unlike the `.github/` exclusion OS-1
removed, this one is deliberate policy. It is not part of the repository, so it
cannot be part of a repository diff, and no acceptance criterion may depend on
it.

`README.md` was checked and publishes **no throughput figures** — only a
description of what the benchmark runner measures. There is therefore no
in-repository published number requiring correction.

The correction obligation is preserved in two places that *are* in the
repository: the superseding ADR (Scope item 9), and a non-diff author action
recorded in `tasks.md` alongside OS-1's branch-protection precedent. DEC-049
calls the `article-final.md` correction "still required"; that requirement stands
as an author obligation and is not weakened here — it is simply not verifiable by
a reviewer reading the repository, and this change says so rather than pretending
otherwise.

## Compatibility invariants

Mandatory implementation review checks. Every one of these must remain true. Any
violation is a defect, not a trade-off.

**C1.** The non-streaming path is byte-identical. `ChatCompletionResponse` gains
no field, loses no field, and `ChatService.complete` is unchanged.

**C2.** A streaming client that sends no `stream_options` receives content events
and `data: [DONE]` as before, plus `"finish_reason"` keys. It receives no usage
event.

**C3.** `data: [DONE]` remains the final event on every path, including the error
path, exactly once.

**C4.** `playground/streaming.py` continues to work unmodified. It returns on
`data: [DONE]` and skips lines yielding no token, so terminal and usage events are
ignored safely.

**C5.** `playground/app.py` is not modified, and its displayed token counts are
unchanged.

**C6.** The stream timeout path (`INFERENCE_X_STREAM_TIMEOUT_S`) emits the same
`data: {"error": ...}` event followed by `data: [DONE]`.

**C7.** `EngineDriver`'s dead-flag semantics, locking, submission rejection after
death (DEC-043), and restart policy are unchanged. `test_engine_driver.py` passes
under `--count=20` as before.

**C8.** Admission control behaviour is unchanged. `routing/admission.py` is not
modified, and `max_tokens` clamping continues to apply before dispatch.

**C9.** `EnginePool`, `api/deps.py`, and the composition root are unchanged. No
route handler imports a concrete engine.

**C10.** Non-streamed `usage` values are unchanged for identical inputs — OS-2
makes the streamed path agree with the non-streamed one, not the reverse.

**C11.** The `/v1/metrics` response schema is unchanged. Fields may be `None`
where they previously held an estimate; no field is added or removed.

**C12.** `vllm` remains a required dependency. No `inference_x/execution/`
package. No second backend. `ModelEntry.engine` remains `Literal["vllm"]`.

**C13.** The DEC-048 mypy baseline does not grow. Adding a module to it is a
review finding.

**C14.** `docs/PHASES.md` is untouched; the phase-namespace question stays open.

## Intentional behavioral changes

Exhaustive. Nothing outside this list may change.

**B1.** SSE content events gain a `"finish_reason": null` key.

**B2.** A terminal choice event with `"delta":{}` and a non-null `finish_reason`
is emitted before `[DONE]` on the success path. Previously no such event existed.

**B3.** A usage event with `"choices":[]` and a populated `usage` object is
emitted before `[DONE]` when `stream_options.include_usage` is true.

**B4.** `ChatCompletionRequest` accepts a new optional `stream_options` object.
Absent means `include_usage: false`.

**B5.** `/v1/metrics` completion-token and tokens-per-second figures for streamed
requests now derive from engine accounting when a usage event is present, and are
**recorded as absent** when it is not. They are never estimated. This is a
deliberate loss of coverage in exchange for correctness — see R2.

**B6.** Benchmark `tokens_generated` and `tokens_per_sec` for streamed prompts
change value, because they were wrong. Previously stored benchmark results are not
comparable with new ones.

**B7.** `BaseEngine.generate_stream`'s return type changes. Any third-party
implementation of `BaseEngine` breaks. There are none outside this repository.

**B8.** `EngineDriver.submit_stream`'s queue element type widens.

## Repository impact

**Modified — runtime**

| Path | Ownership change |
|---|---|
| `schemas/chat.py` | Wire vocabulary. Gains `ChatStreamChunk` and `stream_options`. |
| `engines/base.py` | **The Engine Boundary surface.** Widened return type; docstring corrected. The only DEC-047-governed change in Phase A. |
| `engines/driver.py` | Stream channel carries terminal metadata. Structure, locking, and lifecycle untouched. |
| `engines/vllm_engine.py` | Implements the widened contract. |
| `services/chat_service.py` | Owns SSE serialization and ordering. |
| `observability/middleware.py` | Loses its private estimator; becomes a consumer of engine-reported truth. |
| `benchmarks/runner.py` | Stream measurement reads `usage`. |

**Modified — tests**

Classified by what actually changes, so no file gets a churn edit it does not
need:

| File | `generate_stream` stubs | Change required |
|---|---|---|
| `test_chat_service.py` | 2 | Stubs **and** SSE-sequence assertions |
| `test_routes.py` | 3 | Stubs; one SSE assertion |
| `test_observability.py` | 3 | Stubs; recorded-metric assertions |
| `test_startup.py` | 2 | Stubs only |
| `test_engine_interface.py` | 1 | Stub and its chunk-shape assertion |
| `test_engine_pool.py` | 1 | Stub only |
| `test_vllm_engine_stream.py` | 0 | **Assertions only** — it exercises the real `VLLMEngine.generate_stream`, so it asserts the new chunk shape. No stub to update. |
| `test_streaming.py` | 0 | **Addition only** — it tests `playground/streaming.py`, which C4 keeps unmodified. Gains the C4 tolerance test (task 8.6); its existing assertions stay as they are. |

New test files may be added for the authoritative streaming contract test if that
reads better than extending `test_chat_service.py`.

**Modified — docs**

`docs/DECISIONS.md` — accept DEC-049; add the superseding ADR.

**Not modified**

`routing/`, `api/`, `playground/`, `config/`, `benchmarks/advisor.py`,
`benchmarks/schemas.py`, `benchmarks/prompts/`, `docs/PHASES.md`,
`docs/ARCHITECTURE.md`.

## Acceptance criteria

### Metric truth

1. `grep -rn '\.split()' src/ --include=*.py` returns **nothing**. Verified at
   authoring time: `src/` contains exactly two `.split()` call sites, and both are
   token approximations —
   `observability/middleware.py::_count_sse_delta_tokens` and the
   `benchmarks/runner.py` stream loop. There is no legitimate `.split()` use in
   `src/` for this criterion to trip over, so it is executable as written.
2. For the same prompt and parameters, a streamed request with
   `include_usage: true` and a non-streamed request report the same
   `completion_tokens`.
3. Streamed `usage` values originate from engine accounting, using the same
   derivation `generate()` uses.
4. When `include_usage` is false, no completion-token figure is recorded for that
   request — not a zero, not an estimate.

### Contract

5. The event ordering in R3 is exactly as specified, verified by an assertion over
   the full event sequence, not by substring matching.
6. `data: [DONE]` is last and singular on the success path, the
   `include_usage: true` path, and the timeout path.
7. `ChatStreamChunk` carries exactly the three fields in Scope item 1.

### Boundary

8. `generate_stream` yields `ChatStreamChunk` on `BaseEngine` and every
   implementation and stub.
9. `inference_x/execution/` does not exist. No backend-neutral chunk type exists
   outside `schemas/`.
10. `engines/base.py`'s change is recorded in an accepted ADR (DEC-049).
11. `count_prompt_tokens` is **not** declared on `BaseEngine` — that is OS-4.

### Repository

12. Every Compatibility Invariant C1–C14 holds.
13. Only behaviours B1–B8 changed.
14. The DEC-048 mypy baseline has not grown. If `services/chat_service` now
    type-checks, removing it from the baseline is permitted and preferred; adding
    anything is not.
15. `ruff check .` and `mypy src/` pass; the full unit suite passes.
16. `openspec validate --strict` passes for this change.

## Validation

1. **Streaming contract test — authoritative.** A test that drives
   `ChatService.stream_response` and asserts the **complete ordered event
   sequence** for three cases: default (no `stream_options`),
   `include_usage: true`, and the timeout path. It must assert order and event
   count, not merely that certain strings appear. This test is the protocol
   specification in executable form and must not be weakened to accommodate an
   implementation that is easier to write.
2. **Streamed/non-streamed agreement test.** Same prompt and parameters through
   both paths; assert equal `completion_tokens`.
3. **Absence test.** With `include_usage` false, assert the recorded metric has no
   completion-token figure — specifically that it is absent rather than zero.
4. **Driver regression.** `uv run pytest tests/unit/test_engine_driver.py --count=20`
   — all runs pass, confirming C7.
5. **Consumer compatibility.** `playground/streaming.py` parses a stream
   containing terminal and usage events without modification (C4).
6. **Full suite, lint, types.** `uv run pytest tests/unit -q`,
   `uv run ruff check .`, `uv run mypy src/`.
7. **Invariant sweep.** Confirm `git diff --stat` touches no file outside
   Repository Impact.

## Rollback

Revert the commit. There is no migration, no persisted schema change, and no
stored data whose shape depends on this change.

Two items outlive a revert:

- **The superseding ADR stays.** Previously reported streamed counts were always
  wrong; reverting the fix does not make them right. If OS-2 is reverted, add a
  follow-up ADR explaining the reversion rather than deleting the supersession
  (`docs/PHASE-A-EXECUTION-PLAN.md` §8.2).
- **Benchmark results produced under OS-2 remain incomparable** with pre-OS-2
  results in both directions. Reverting reintroduces the approximation; it does
  not restore comparability.

Partial rollback is not supported. The chunk model, the driver channel, and the
SSE serialization are one contract; reverting one leaves the engine unable to
satisfy its own interface.

## Dependencies

**Upstream:** OS-1 (complete and enforced). DEC-049 must be `accepted` before
implementation — it is currently `proposed`. Accepting it is task 0.1.

**Downstream:** OS-4 depends on this change (shares the SSE generator and the
response-construction path). OS-6 depends on this change (shares the
`PromptResult` construction in `benchmarks/runner.py`).

**Parallel-safe:** OS-3 and OS-5 may proceed concurrently. OS-5 touches
`benchmarks/runner.py::_load_suite`, disjoint from the stream loop this change
edits; whichever opens second rebases before review.

**External:** none.

## Risks

**1. The Engine Boundary widening is relitigated during review.** *(Likelihood:
medium-high. Impact: high — stalls the Phase A critical path.)* DEC-049 decides
it, R1 fixes the name, and Scope item 1 fixes the field list. A reviewer proposing
a chunk type outside `schemas/` is proposing the `inference_x/execution/`
prohibition under another name; the answer is no. Do not reopen DEC-047 or
DEC-049 in this change.

**2. Driver changes destabilise the DEC-043 race fix.** *(Likelihood: medium.
Impact: high.)* `engines/driver.py` carries a fix for a race that reproduced ~40%
of the time. Widening the queue element type touches `_dispatch` and `_fail_one`.
*Mitigation:* C7, validation step 4 (`--count=20`), and a hard constraint against
touching locking or lifecycle.

**3. Metric coverage loss is mistaken for a bug.** *(Likelihood: medium. Impact:
medium.)* After this change, default streaming requests contribute no
completion-token figure to `/v1/metrics`. Someone will read that as a regression
and "fix" it by always emitting usage or by reinstating an estimate. *Mitigation:*
R2, B5, and acceptance criterion 4 state it as intended.

**4. Scope creep into OS-4 via the SSE generator.** *(Likelihood: medium. Impact:
medium.)* Both units restructure the same generator. Adding `warnings` or
`resolved` "while we're here" is the failure mode. *Mitigation:* R4 and the
non-goals.

**5. Test-stub churn hides a contract violation.** *(Likelihood: medium. Impact:
medium.)* Eight test files change mechanically. A stub updated to yield whatever
makes its assertion pass can mask a real mismatch. *Mitigation:* validation step 1
tests the real service against the real contract, not stubs against stubs.

**6. Silent observability failure.** *(Likelihood: low. Impact: medium.)* The
middleware swallows exceptions by design. A parsing mistake in the repointed
recorder produces no error, just missing data. *Mitigation:* acceptance criteria 2
and 4 assert recorded values directly.

## Future follow-ups

Recorded so they are not smuggled in. None is authorized by this change.

- **Playground token counts.** `playground/app.py` retains two
  `len(content.split())` sites for its own display. Correcting them means
  consuming the usage event in the TUI and setting `include_usage` on playground
  requests. Client-side work, separate change.
- **`article-final.md` correction.** Author obligation under DEC-049, outside the
  repository (R5).
- **DEC-048 baseline reduction.** If `services/chat_service` type-checks after the
  widening, removing it is permitted here; broader reduction is separate.
- **Per-request decomposed timings.** Phase B3 will want to widen
  `ChatStreamChunk`. Deliberately not anticipated now.
- **`error` as a streamed `finish_reason`.** The error path keeps its current
  shape (R3). Making error termination structured is a wire-contract change of its
  own.
- **`engines/registry` factory.** Phase A/B seam, tracked from DEC-049 and
  `docs/PHASE-A-EXECUTION-PLAN.md` §5.4.
