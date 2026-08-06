# Design: expose-per-request-engine-timing

## Context

B2 (`expose-native-engine-metrics`) established that vLLM's `AsyncLLM`
already tracks rich internal state and that mounting it is a passthrough,
not a reimplementation. That change operated at process scope
(`GET /metrics`, scraped independently of any request). This change asks
the same question one level down: does `AsyncLLM` already track *per-request*
timing InferenceX can attach to that request's own response, without
reimplementing timing measurement itself?

Per the task's explicit requirement, nothing below is inferred from prior
reviews or from B2's document. Every field, every clock domain, and every
formula was re-verified directly against the installed vLLM 0.22.1 in this
repository's own `.venv`, both by reading source and by executing a real
`AsyncLLM.generate()` call and inspecting the live object it returns.

## Verification log (empirical, vLLM 0.22.1)

### 1. Where per-request timing lives

`vllm.outputs.RequestOutput.__init__` takes a `metrics` parameter typed
`vllm.v1.metrics.stats.RequestStateStats | None`. Source
(`vllm/v1/metrics/stats.py`):

```python
@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""
    num_generation_tokens: int = 0
    # This is an engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0
    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    # first token latency
    first_token_latency: float = 0.0
    is_corrupted: bool = False
```

The comments quoted above are vLLM's own, not paraphrased — they are the
basis for the clock-domain finding in §3.

### 2. It is genuinely populated for InferenceX's configuration

`vllm/v1/engine/output_processor.py`:
`self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None`
— gated on `log_stats`, which is `not disable_log_stats`. `VLLMEngine.__init__`
(`engines/vllm_engine.py:424`) constructs `AsyncEngineArgs(**kwargs)` and
never sets `disable_log_stats`; `AsyncEngineArgs.disable_log_stats` defaults
to `False`. Confirmed live: `AsyncEngineArgs(...).disable_log_stats == False`.
This same `self.stats` object is attached verbatim to every `RequestOutput`
yielded to the caller (`output_processor.py:373`, `metrics=self.stats`) —
it is not a separate internal-only structure InferenceX would need a new
code path to reach.

**Live execution, not just source reading:** constructed a real `AsyncLLM`
against `Qwen/Qwen2.5-0.5B-Instruct` (the same model InferenceX serves) via
`AsyncLLM.from_engine_args(AsyncEngineArgs(...))`, exactly mirroring
`VLLMEngine.__init__`'s construction, and ran a real
`llm.generate(prompt=..., sampling_params=..., request_id=...)` call end to
end. Every yielded `RequestOutput.metrics` was a populated
`RequestStateStats`, not `None`, on every chunk including the terminal one:

```
arrival_time=1786012425.083239 queued_ts=187239.580166 scheduled_ts=187239.580180
first_token_ts=187240.235890 last_token_ts=187240.331037 first_token_latency=0.658468
num_gen_tokens=20 is_corrupted=False
```

### 3. Clock domains — verified, not assumed

`arrival_time` is set via `time.time()` in the **frontend** process
(`vllm/v1/request.py:95`, `vllm/v1/engine/async_llm.py:742`) — wall-clock,
epoch-based (consistent with the ~1.786×10⁹ value observed above).

`queued_ts` / `scheduled_ts` come from `EngineCoreEvent.timestamp`
(`vllm/v1/engine/__init__.py`), whose own docstring states: *"The timestamp
is a monotonic timestamp... These timestamps should not be compared with
timestamps from other processes."* `EngineCoreEvent.new_event()` stamps it
with `time.monotonic()`. This event is created and timestamped inside the
**engine-core process** (`vllm/v1/core/sched/scheduler.py`), consistent
with the observed value (~187239s, i.e. time-since-boot, not epoch time).

`first_token_ts` / `last_token_ts` come from `engine_core_timestamp`, traced
to `EngineCoreOutputs.timestamp`
(`vllm/v1/engine/__init__.py`, `__post_init__`: `self.timestamp =
time.monotonic()` if unset) — also stamped in the **engine-core process**,
on the output batch before it crosses the IPC boundary to the frontend.

**Conclusion:** `queued_ts`, `scheduled_ts`, `first_token_ts`, and
`last_token_ts` share one clock domain — `time.monotonic()`, read inside
the engine-core process. `arrival_time` is a different domain — `time.time()`,
read in the frontend process. Subtracting any of the first four from each
other is safe. Subtracting `arrival_time` from any of them is not verified
safe and is not done anywhere in this design.

This distinction is not theoretical: vLLM's own `first_token_latency`
(`vllm/v1/metrics/stats.py:369`, `self._time_since(req_stats.arrival_time)`)
is computed as `IterationStats.iteration_timestamp - arrival_time`, and
`iteration_timestamp` is itself `time.time()`
(`vllm/v1/metrics/stats.py:329`) — i.e. `first_token_latency` is a
**same-domain, frontend-side wall-clock** measurement, safe in its own
right, but a different measurement from the engine-core-only intervals
below. It is excluded from this change anyway (see proposal.md, Out of
scope) because it duplicates `GET /v1/metrics`'s existing TTFT figure
(DEC-049), not because it is unsafe.

The observed ~2.9 ms gap between `first_token_latency` (0.658468 s) and
`first_token_ts - scheduled_ts` (0.655710 s) in the verification run is
consistent with this: the former includes frontend-to-engine-core IPC
round-trip overhead the latter does not.

### 4. Derived intervals — mirrored from vLLM's own formula, not invented

`output_processor.py`'s `do_tracing()` (vLLM's own OpenTelemetry span
builder, using the identical `RequestStateStats` object) computes:

```python
queued_time = metrics.scheduled_ts - metrics.queued_ts
prefill_time = metrics.first_token_ts - metrics.scheduled_ts
decode_time = metrics.last_token_ts - metrics.first_token_ts
inference_time = metrics.last_token_ts - metrics.scheduled_ts
```

This change reuses these exact four formulas (renamed to InferenceX's
`_ms` convention; see §5). They are not this design's invention — they are
vLLM's own internal composition of its own timestamps, verified live:

```
queued_time (scheduled-queued):   1.405e-05 s   (~0 — GPU was idle)
prefill_time (first_token-sched): 0.655710 s
decode_time (last_token-first):   0.095148 s
inference_time (last_token-sched):0.750857 s     (= prefill + decode, confirmed)
```

### 5. What is explicitly excluded, and why

| Field | Verified? | Excluded because |
|---|---|---|
| `arrival_time` | Yes, populated | Different clock domain (frontend wall-clock); not comparable to the four engine-core fields without a cross-process assumption this design does not make. |
| `first_token_latency` | Yes, populated | Same-domain and safe, but duplicates `GET /v1/metrics`'s existing TTFT (DEC-049) — would blur the "engine timing vs. HTTP timing" line this change is required to keep clean. |
| `num_generation_tokens` (on `RequestStateStats`) | Yes, populated | Duplicates `ChatCompletionUsage.completion_tokens`, already reported by `derive_terminal_metadata`. |
| `is_corrupted` | Yes, populated (`False` in verification) | A NaN-logit corruption flag, not a timing signal — out of scope for this change. |
| A second "scheduler delay" distinct from "queue wait" | **No** — does not exist | vLLM 0.22.1 records exactly one `QUEUED`→`SCHEDULED` event pair per (non-preempted) request. Inventing a second field here would violate the task's explicit "do not invent fields" requirement. |

### 6. Stability: public attribute, internal dataclass

`RequestOutput.metrics` is a documented constructor parameter on
`vllm.outputs.RequestOutput` — vLLM's standard, top-level, stable output
type, the same type `AsyncLLM.generate()` has always yielded. The
**attribute's existence** is as stable as `RequestOutput` itself.

The **value type**, `vllm.v1.metrics.stats.RequestStateStats`, is not
re-exported from top-level `vllm` and lives under the `v1` internal
namespace with an engineering-voice docstring ("Stats that need to be
tracked across delta updates"), not a public-API docstring. vLLM makes no
documented stability commitment on this dataclass's field set. This design
treats it accordingly: every read is defensive (see Decisions below), and
the compatibility invariants document exactly what breaks and how it
degrades if a future vLLM release renames or removes a field.

## Goals / Non-Goals

**Goals**

- Surface the four timing intervals vLLM 0.22.1 already computes
  internally, per request, as an additive response field.
- Keep engine timing and HTTP-boundary timing unmistakably separate — no
  shared field name, no shared computation, no mixed clock domain.
- Reuse the existing DEC-050 single-source call site
  (`derive_terminal_metadata`) rather than adding a second read of engine
  state.

**Non-Goals**

- Replacing or aggregating with `GET /metrics` (B2) or `GET /v1/metrics`.
- A general OpenTelemetry tracing integration (vLLM has one internally,
  gated on `trace_headers`; this change does not wire it up or depend on
  it).
- Solving preemption-aware timing, multi-model pool timing attribution, or
  any B4/B5/B6 concern.

## Decisions

### Decision 1 — `EngineTiming` is a new `schemas/chat.py` model, not a reuse of `RequestStateStats`

`RequestStateStats` is vLLM-internal (§6) and carries fields this change
deliberately excludes (`arrival_time`, `first_token_latency`,
`num_generation_tokens`, `is_corrupted`). Returning it directly would leak
an unstable vLLM-internal type through the Engine Boundary and expose
fields this design explicitly decided against. `EngineTiming` is a small,
InferenceX-owned Pydantic model with exactly the four derived
milliseconds fields — the same pattern `ChatCompletionUsage` already
established for `usage`.

### Decision 2 — Fields are named `*_time_ms`, matching the existing `_ms` convention

`GET /v1/metrics`'s `MetricsResponse` already uses `avg_latency_ms`,
`p95_latency_ms`, `avg_ttft_ms`. `EngineTiming` follows the same units and
suffix convention: `queue_time_ms`, `prefill_time_ms`, `decode_time_ms`,
`inference_time_ms`. `inference_time_ms` is arithmetically
`prefill_time_ms + decode_time_ms`; it is exposed directly (mirroring
vLLM's own `do_tracing()`, which reports all four independently) rather
than requiring the client to re-derive it.

### Decision 3 — Derivation extends `derive_terminal_metadata`, not a new function

`derive_terminal_metadata(output) -> tuple[finish, usage]`
(`engines/vllm_engine.py:36`) is the one place `_stream_chunks` reads a
finished `RequestOutput`, and DEC-050 exists specifically to guarantee
streaming and non-streaming cannot report different `usage` for the same
request. The same guarantee must hold for timing, for the same reason —
so timing is derived at the same call site, from the same `output`, in the
same function. The return type widens to
`tuple[finish, usage, EngineTiming | None]`.

### Decision 4 — Absence over estimation, matching DEC-049's precedent exactly

When `output.metrics` is `None` (a future engine/config with
`disable_log_stats=True`, or a future non-vLLM `BaseEngine`), `timing` is
`None` on the response — never a zero, never an estimate. This is not a
new policy; it is DEC-049's "Absence replaces estimation" rule for `usage`,
applied to the same call site for the same reason (§4 of DEC-050:
"consistent with this codebase's existing posture for advisory/best-effort
signals").

### Decision 5 — Defensive attribute access, given §6's stability finding

Because `RequestStateStats` is an internal vLLM type with no documented
field-stability guarantee, the derivation reads its four fields via
`getattr(..., default=None)` rather than direct attribute access, and
treats any missing/non-numeric field as equivalent to `output.metrics is
None` (whole-`timing` absence, not a partially-populated `EngineTiming`).
A partially populated timing block would be worse than none: it would
imply completeness that isn't there.

### Decision 6 — On the wire, `timing` rides the existing usage event, not a new one

`ChatService.stream_response()` (`services/chat_service.py`) does not
serialize `ChatStreamChunk` directly — it hand-builds SSE JSON payloads and
already splits one engine-level terminal `ChatStreamChunk` (which carries
`finish_reason`, `usage`, and now `timing` together) into two wire events:
a `finish_reason`-only event, and a separate `usage`-only event emitted
only when `stream_options.include_usage` is set (DEC-049). Adding a second,
independent gate for `timing` would be a new request-facing flag, which
proposal.md rules out ("no changes to request semantics"). Instead,
`timing` is added to the existing usage event's payload, gated by the same
`include_usage` flag — one opt-in for all terminal metadata a client didn't
ask to receive by default, not two. Non-streaming `ChatCompletionResponse`
needs no equivalent decision: it is a single Pydantic response model,
serialized whole, so `timing` (like `usage`) is simply always present when
the engine supplied it.

## Compatibility Invariants

1. `ChatCompletionResponse.timing` and `ChatStreamChunk.timing` are strictly
   additive `Optional` fields, default `None`. No existing field, type, or
   required-ness changes. Existing clients and tests that do not reference
   `timing` are unaffected. On the wire, `ChatService.stream_response()`
   places `timing` on the same SSE event as `usage` — gated by the same
   `stream_options.include_usage` opt-in, not a new flag (see Decision 6).
2. Engine timing is computed exclusively from `RequestStateStats`'s four
   engine-core-process monotonic fields (`queued_ts`, `scheduled_ts`,
   `first_token_ts`, `last_token_ts`). It is never computed from, or mixed
   with, `arrival_time`, any HTTP-boundary timestamp
   (`observability/middleware.py`), or any wall-clock value.
3. `derive_terminal_metadata` remains the single call site deriving
   terminal metadata from a finished `RequestOutput` (DEC-050); this
   change widens its return value, it does not add a second reader of
   engine state.
4. Non-streaming `generate()` and streaming `generate_stream()` report
   identical `EngineTiming` for the same request, because both derive from
   the same `derive_terminal_metadata` call (DEC-050 extended, not
   re-litigated).
5. `timing` is `None` whenever `output.metrics` is `None` or any of the
   four required fields cannot be read — never a partial or estimated
   value.
6. No change to `GET /metrics` (B2) or `GET /v1/metrics`
   (`observability/`) — neither their schemas, computations, nor
   exclusion rules are touched.
7. `ResolvedRequest` is unchanged — this is response data describing what
   happened, not a request parameter describing what was asked for; it
   does not belong to the Effective Request block (`schemas/chat.py`
   `ResolvedRequest` docstring's own scoping rule).

## Risks / Trade-offs

- **`RequestStateStats` is an unstable internal type (§6).** A future vLLM
  release could rename or remove a field. Mitigated by Decision 5
  (defensive `getattr`, whole-block absence on any read failure) and by
  this change being additive — a future vLLM upgrade that breaks the read
  degrades `timing` to always-`None`, it does not fail requests.
- **Preemption is invisible to `queue_time_ms`.** vLLM's own scheduler
  comment (`if req_stats.scheduled_ts == 0.0: # ignore preemptions`) means
  a request preempted after its first scheduling keeps its original
  `scheduled_ts`. This change reports vLLM's number as-is; it does not
  reconstruct or flag preemption history (see proposal.md, Out of scope).
- **Two additive optional fields on two response types is small surface,
  but still surface.** Accepted: this is the same shape of change B2 was
  (additive, opt-in-by-presence, zero impact on non-consumers), and is the
  reservation `ChatStreamChunk`'s own docstring already made for Phase B3.

## Validation properties (required)

- `ruff check .`, `mypy src/`, `pytest tests/unit` all green.
- A live-verified test (mirroring this design's own verification
  methodology) confirming, against the pinned vLLM 0.22.1, that
  `RequestOutput.metrics` is populated for a real `generate()` call and
  that `inference_time_ms == prefill_time_ms + decode_time_ms` within
  floating-point tolerance.
- Non-streaming and streaming paths tested to report identical
  `EngineTiming` for the same request (extends the existing DEC-050
  usage-parity test).
- A unit test forcing `output.metrics = None` (or a missing field) asserts
  `timing` is `None`, not a partial or zero-filled `EngineTiming` —
  extends the existing DEC-049 usage-absence test pattern.
- `openspec validate expose-per-request-engine-timing --strict` passes.
