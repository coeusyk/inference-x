# Design — OS-4 truthful runtime resolution

Implementation mechanics for the frozen architecture. No future architecture is
discussed here; where a decision was already made by an ADR, this document says
which one and moves on.

## Ownership

| Concern | Owner | Not the owner |
|---|---|---|
| Deciding a substitution | `AdmissionController.admit()` | `ChatService`, engines |
| Producing a `ResponseWarning` | `AdmissionController.admit()` | `ChatService` |
| Raising under `strict` | `AdmissionController.admit()` | the route handler |
| Building the Effective Request | `ChatService` (already does) | admission |
| Serializing `resolved` to the wire | `ChatService` | engines |
| Attaching `resolved`/`warnings` to a response | `ChatService` | engines |
| Supplying a prompt-token count | the engine, via `BaseEngine` | admission |
| Falling back when no count exists | `AdmissionController` | the engine |

The single rule behind the table: **admission owns the truth, `ChatService` owns the
wire, engines own neither.** An engine that constructed `resolved` would have to know
about admission, which the Engine Boundary forbids (DEC-047). C16 asserts it.

## Lifecycle

```
_resolve_engine()          routed_model, engine
      │
      ▼
admit()                    ── decides clamps, skips, estimates
      │                    ── builds warnings
      │                    ── raises StrictModeViolationError if strict and substituted
      ▼
AdmissionResult            effective_max_tokens, reserved_tokens, warnings
      │
      ▼
effective_request          request.model_copy(update={"max_tokens": ...})
      │                    ◄── ANCHOR: everything above is now frozen
      ├──────────────► non-streaming: engine.generate() → attach resolved/warnings
      │
      └──────────────► streaming: emit pre-generation event
                                  → content events
                                  → terminal event
                                  → usage event (optional)
                                  → [DONE]
```

The anchor line is the whole design. Everything determined above it may be emitted
before generation; nothing below it may (DEC-053).

## Effective Request

The object already exists. `chat_service.complete()` and
`chat_service.stream_response()` both build:

```python
effective_request = request.model_copy(update={"max_tokens": admitted.effective_max_tokens})
```

This change names it and serializes it. It does **not** introduce a new object, a
new construction site, or a second copy. Both paths keep their single `model_copy`
call and both derive `resolved` from its result, so the two paths cannot disagree.

`ResolvedRequest` is built from `effective_request`, never from `request` — building
it from the original would report what the client asked for rather than what ran,
which inverts the entire point.

## `resolved` derivability

`ResolvedRequest` mirrors `ChatCompletionRequest` minus two exclusion classes:

- **Content** — `messages`. It is the payload, not a parameter, and echoing up to
  32 KB per response is not a contract, it is a bandwidth bug.
- **Transport and policy** — `stream`, `stream_options`, `strict`. These control how
  the response is delivered and how substitution is handled, not how generation runs.

Everything else is included: `model`, `temperature`, `max_tokens`, `top_p`,
`max_context_tokens`, `max_output_tokens`, `priority`, `seed`.

The rule is enforced by a test, not by review (Validation 8): the test compares
`ResolvedRequest.model_fields` against `ChatCompletionRequest.model_fields` minus the
documented exclusion set, and fails if either schema drifts. A field added to the
request in a later unit therefore forces an explicit decision about `resolved` rather
than silently omitting itself.

`resolved` is a full echo rather than a diff. A diff needs the original to be
interpretable; a full echo is self-contained, and self-contained is what makes the
block re-submittable.

## `ResponseWarning`

```python
class ResponseWarning(BaseModel):
    type: Literal["substituted", "degraded"]
    code: str
    message: str
    field: str | None = None
```

- `type` is the axis a client branches on: *the server ran something different* vs
  *the server could not verify something and proceeded*. Two members, closed set.
- `code` is the stable machine identifier. It is what `strict` keys on and what tests
  assert. The five codes this change introduces are tabled in `proposal.md`; the ADR
  carries the same table so the registry has one home.
- `message` is human text and the only field free to change without a spec change.
- `field` names the affected request field where one exists, `None` for a skipped
  gate that maps to no single field.

Why not `list[str]`: `strict` must reject on exactly the conditions that emit a
`substituted` warning (DEC-052 §2). A string list forces the rejection predicate and
the warning predicate to be written twice, and two encodings of one policy drift.
`warnings` is also a published wire field — `list[str]` → `list[object]` later is a
breaking change for every client reading `warnings[0]` as text.

`AdmissionResult` is a frozen dataclass, so it carries
`warnings: tuple[ResponseWarning, ...] = ()`. `ChatService` converts to a list at the
wire boundary.

## `strict` semantics

DEC-052 is accepted and this change implements it verbatim. Mechanically:

```python
# inside admit(), at the point the clamp is decided — not in a second pass
if effective_output < requested_output:
    warning = ResponseWarning(type="substituted", code=..., field="max_tokens", message=...)
    if request.strict:
        raise StrictModeViolationError(...)
    warnings.append(warning)
```

Three properties follow from writing it this way and would not follow from any other
arrangement:

1. **One predicate, two outcomes.** The condition is evaluated once. There is no
   second `if` that could disagree.
2. **`degraded` never rejects.** The `strict` check is inside the `substituted`
   branch only. A skipped gate appends its warning and falls through — which is
   DEC-047 §4's fail-open posture preserved by construction rather than by care.
3. **Strict never changes an accepted execution.** `admit()` either raises or returns
   the same `AdmissionResult` it would have returned with `strict` false, so no
   downstream code branches on `strict` at all. Validation 3 asserts byte-identical
   output.

`StrictModeViolationError` subclasses `ValueError` and is therefore mapped to a
sanitized 400 by the existing `api/errors.value_error_handler` with no new
registration — the same route `ContextTooLongError` already takes.

## `count_prompt_tokens` contract

Declared on `BaseEngine` as **non-abstract with a default returning `None`**:

```python
def count_prompt_tokens(self, request: ChatCompletionRequest) -> int | None:
    """Prompt token count for admission, or None when this engine cannot count.

    Not abstract: an engine without a tokenizer is a supported state, and callers
    must fail open on None rather than treat it as an error (DEC-047 §4).
    """
    return None
```

Abstract was considered and rejected on two grounds, both mechanical:

- A required method removes "unavailable" as a representable state, and DEC-047 §4
  plus plan §9 A.3 both require preserving it. The `int | None` return is what keeps
  the fail-open path expressible.
- Every stub in `tests/unit/` subclasses `BaseEngine`
  (`test_observability.py`, `test_startup.py`, `test_chat_service.py`,
  `test_seed_determinism.py`, `test_routes.py`, `test_engine_pool.py`,
  `test_engine_interface.py`). An abstract method breaks all of them for no
  behavioural gain, since each would implement `return None`.

`VLLMEngine.count_prompt_tokens` already returns `int` — a valid narrowing of
`int | None`, so no signature change is needed there. Only its docstring changes,
because the sentence "BaseEngine doesn't declare this" becomes false.

Admission changes from probing to calling:

```python
# before: counter = getattr(engine, "count_prompt_tokens", None)
count = engine.count_prompt_tokens(request)
if count is None:
    # chars/4 heuristic + degraded warning
```

`AdmissionController.admit()`'s `engine` parameter narrows from `Any` to `BaseEngine`.
`tests/unit/test_admission.py::_FakeEngine` is currently a plain class and must
subclass `BaseEngine` for the narrowed type to be honest.

`kv_capacity_tokens` stays on `getattr` at `admission.py:212` and
`api/routes/metrics.py:43`, byte-identical. DEC-047 §3 marks it provisional. C9.

## Streaming lifecycle

The pre-generation event, emitted once, immediately after `admit()` returns and
before the engine generator is consumed:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[],
       "resolved":{…},"warnings":[…]}
```

Shape choices, each forced by a verified consumer:

- **`choices: []`** — `benchmarks/runner.py:75` reads
  `(chunk.get("choices") or [{}])[0].get("delta", {})`, which resolves an empty list
  to `{}` and yields an empty `content` that the next line skips. The usage event
  already established this shape.
- **No top-level `usage` key** — `middleware._extract_sse_usage` returns `None` for
  any payload whose `usage` is not a dict, so this event contributes no token figure.
- **`object: "chat.completion.chunk"`** — the stream stays homogeneous in event type;
  clients that switch on `object` see nothing new.

Emitted unconditionally, not gated on a request flag. Gating it would make plan §9
C.6 unsatisfiable for any client that did not opt in, which is the criterion this
change exists to meet.

The event is constructed by the same `_event()` helper already in
`stream_response()`. No new serialization path.

### The middleware consequence

`observability/middleware.py:264` currently sets TTFT on the first raw chunk:

```python
if ttft_ms is None:
    ttft_ms = (time.perf_counter() - start) * 1000
```

With a pre-generation event, that timestamps admission rather than first token, and
every `/v1/metrics` TTFT silently becomes incomparable with figures recorded before
this change. The fix anchors the measurement to the first event carrying non-empty
`delta.content`, which is what `benchmarks/runner.py:77` already does — its comment
reads "TTFT still comes from the first content event", so this aligns the two
consumers rather than inventing a rule.

This is preservation of an existing metric's meaning, not Phase B3 timing work. No
timing field is added to any schema and no new figure is recorded.

## Pre-generation metadata phase

Governed by DEC-053. An event belongs here iff every field it carries is fully
determined **and immutable** at the moment `effective_request` is constructed.

What qualifies today: the resolved parameters, the effective seed, the routed model,
and every admission warning — all fixed by `admit()` returning.

What does not, and why the rule is anchored where it is: queue and wait time is
determined when the engine begins processing, which is *after* the anchor but
*before* the first token. Anchoring on "before the first token" would classify it as
pre-generation while the event is already on the wire, making it un-emittable in the
slot the rule assigns it. Anchoring on the effective request excludes it cleanly and
keeps the Phase B3 timing class intact.

Cardinality is exactly one. `resolved` and `warnings` ride the same event rather than
two, because both are determined at the same instant and splitting them would create
an intra-phase ordering contract for no gain.

## Post-generation metadata phase

Events after the terminal event carry only what generation produced. The usage event
is last before `[DONE]`.

This is not new behaviour — OS-2 already emits exactly that sequence — but it is now
a normative requirement rather than an implementation detail, so that the ordering
cannot be extended from the tail by a later change. C3 asserts it on every path.

The error path is unchanged from DEC-049: on timeout the service emits its error
event and `[DONE]`, with no terminal and no usage event. The pre-generation event has
already been emitted by then, which is a property worth having rather than a side
effect — it is the path on which knowing what the server resolved matters most.
