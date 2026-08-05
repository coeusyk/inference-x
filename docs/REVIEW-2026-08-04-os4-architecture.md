# OS-4 — Principal Engineer Architecture Review (pre-OpenSpec)

Date: 2026-08-04 · Branch: `develop` @ 99a0c12 · Verdict: **APPROVE WITH MODIFICATIONS**

Reviewed against repository state as authoritative: `docs/PHASE-A-EXECUTION-PLAN.md`
§4/§9, `openspec/specs/platform/spec.md`, `docs/DECISIONS.md` DEC-047/049/050/051,
`src/inference_x/services/chat_service.py`, `routing/admission.py`,
`schemas/chat.py`, `engines/base.py`, and the archived OS-2 proposal.

---

## 0. The finding that survives every branch

The proposed ordering

```
content → terminal → usage (optional) → resolution → [DONE]
```

places an event **after** the usage event. OpenAI documents the usage chunk as
*the additional chunk streamed before `data: [DONE]`*, and clients in the wild
rely on it as the end-of-stream sentinel — `usage != null` is treated as "this is
the last thing before DONE." Inserting a further event between them contradicts
the documented reading and will be dropped, or will terminate parsing early, on
clients that sentinel this way.

Every other question in this review is a judgment call. This one is a defect in
the proposal as stated. **Whatever else is decided, nothing is emitted after the
usage event.**

---

## 1. Architectural strengths

**S1 — The concern is correctly scoped to one sentence.** "The client can see
every substitution the server made" is a contract property, not a feature list.
It gives the unit a falsifiable boundary: anything that is not a substitution the
server made does not belong in OS-4. That is why the unit has survived three
plan revisions without drifting.

**S2 — Degradation gets one mechanism with two sinks.** Plan §3.4(c) — structured
log (DEC-047 §4) plus response `warnings` — is right and non-obvious. Two
independent mechanisms would guarantee the second rewrites the first, and would
let the log and the wire disagree about what happened. One predicate, two
serializations, is the only version of this that stays true.

**S3 — The durable/provisional split is held correctly.** `count_prompt_tokens`
declared on `BaseEngine`; `kv_capacity_tokens` left on `getattr`. This is the
rare case of a codebase resisting the urge to make its capability surface
uniform. Uniformity here would freeze a provisional number (DEC-047 §3) as a
cross-backend contract that Phase B4 may delete. Keeping two discovery mechanisms
is *more* honest than one, and the plan says so explicitly.

**S4 — Fail-open is preserved deliberately, not by omission.** DEC-047 §4 forbids
tightening admission to fail-closed; OS-4 makes existing degradation observable
without changing what admission decides. Observability-before-enforcement is the
correct sequencing: you cannot responsibly tighten a gate whose skip rate you
have never measured.

**S5 — The effective request already exists in the code.** Both `complete()` and
`stream_response()` build `request.model_copy(update={"max_tokens": ...})`. OS-4
is naming an object the system already constructs, not inventing an abstraction.
That is the strongest possible position from which to add a wire contract.

---

## 2. Architectural concerns

### C1 — §4 and §9 C.6 disagree about whether streaming carries `resolved`

Both are repo-authoritative and they conflict:

- **§4 "Contains"** — `warnings` and a `resolved` block **on the response**;
  silent on streaming. `ChatCompletionResponse` is the non-streaming body.
- **§9 C.6** — "A client can determine, **from the response alone**, every
  parameter the server substituted." For a streamed request, the response *is*
  the event stream.

This is an internal ambiguity in the plan, not scope creep in the proposal. It
must be resolved before the OpenSpec is authored, because the delta shape depends
on it: with the carrier in, OS-4 **modifies** the live requirement
`openspec/specs/platform/spec.md` → "OpenAI-compatible streaming event order";
with it out, OS-4 touches no streaming requirement at all. Getting that wrong
makes the spec history lie about when the event order changed.

### C2 — A dedicated event is right; a trailer position is wrong

On the abstraction question the proposal is correct: a **dedicated event beats
extending the terminal metadata event.** Overloading the terminal chunk gives you
an event meaning "generation ended, here is why" that also carries "here is what I
substituted before generation began" — two lifecycles on one event. Worse, the
terminal chunk carries `choices[0]`; a top-level `resolved` on an event that also
carries a per-choice payload mixes per-request and per-choice semantics, which
breaks the moment `n > 1` is supported. A separate event with `choices: []`
follows the precedent OS-2 already set with usage.

The position is the defect. Resolution is **fully determined at admission**,
before token 1 — `admit()` returns before `generate_stream` is called. Delivering
a pre-generation fact as a trailer costs three things:

1. **It is lost on the error path.** On stream timeout the service emits its error
   event and `[DONE]`, with no terminal and no usage event. A trailing resolution
   event is lost on exactly the path where "your `max_tokens` was clamped 4096→512"
   is most diagnostically valuable.
2. **The client cannot act on it.** A client that would abort on a heavy clamp
   must first consume the entire generation it no longer wants.
3. **It collides with Phase B.** Timings (Phase B3) are *only* knowable at the
   end. If resolution occupies the trailer slot, Phase B arrives wanting the same
   slot for genuinely-terminal data, and you get two competing trailers.

That third point is the 3–5 year question. There are two kinds of out-of-band
stream metadata and they should never share a slot:

| | Known when | Contents | Natural slot |
|---|---|---|---|
| **Pre-generation** | at admission | resolved params, effective seed, routed model, admission warnings | prologue |
| **Post-generation** | at completion | usage, TTFT/queue timings (B3), replay handle / trace id (C) | the existing usage/terminal trailer |

Under that split, Phase B timings attach to the usage trailer where they belong,
and Phase C replay divides cleanly: replay *inputs* (seed, resolved params, model
revision) are prologue; a replay *handle* is trailer. The taxonomy does the
routing, so neither phase has to relitigate placement.

**Prologue is the architecturally correct position, but it is not free**, and the
cost lands outside Phase A: a prologue event corrupts any TTFT measurement that
timestamps first-chunk-received. `benchmarks/runner.py` is a first-party consumer
and Phase B3 explicitly owns per-request timings. It is fixable — define TTFT as
first *content* chunk — but that is a constraint this review would be imposing on
a Phase B unit. Note also that OS-2's compatibility check verified both SSE
consumers tolerate an extra **terminal** chunk; tolerance at arbitrary position is
an inference from how they skip (`playground/streaming.py` returns on `[DONE]` and
skips lines yielding no token; the runner skips chunks without `delta.content`),
not something OS-2 verified.

### C3 — "Runtime Resolution" conflates two lifetimes

The name implies a runtime that decides things. But §4's `Contains` list sources
the block from two places with different lifetimes:

- **Per-request** — `AdmissionController` clamps, effective seed, routed model.
  Decided per call.
- **Per-process** — `apply_tier_knobs` caps. Decided once at engine construction
  (`utils/vllm_pool_config.py`, Phase B6/D-owned). Identical for every request in
  the process lifetime.

A block that mixes them tells the client "here is what was resolved for *your
request*" while some entries are facts about the server. Over time this is how
`resolved` becomes the junk drawer for anything the server wants to confess.

I am contesting a line in §4's `Contains` list here, not describing it. The
resolution that satisfies §4's intent without the lifetime conflation: **a tier
cap that caused a clamp surfaces as the clamp warning's reason, not as a
`resolved` field.** The client learns the cap existed and that it bit — which is
what §4 wants — without `resolved` acquiring process-level state. Static server
configuration belongs on `/v1/metrics` or a capabilities endpoint.

**"Effective Request" is the abstraction that ages better**, and the rename is
free because the object already exists in `chat_service.py`. Its value is a
derivability rule:

> A field appears in `resolved` **iff** it exists on `ChatCompletionRequest`.

That rule makes the block's contents derivable rather than negotiable — it
pre-answers every future "should X go in `resolved`?" argument. Timings: not a
request field, excluded automatically. Seed, sampling params, model revision:
request-shaped, included naturally. And it produces the Phase C forcing argument
for free — **replay becomes `POST /v1/chat/completions` with `resolved` as the
body.** No re-derivation, no parallel replay schema. A curated grab-bag cannot
offer that.

Recommendation: name the concept **Effective Request** in the spec and ADR; keep
the wire field named `resolved` (already pinned to OS-4 by OS-2's R4 table). Zero
wire churn, a real invariant.

### C4 — `warnings: list[str]` cannot support strict mode

The decisive argument is §9 C.7: `strict: true` must reject where the default
clamps. That requires warning-generation and the strict predicate to be **one code
path** — the condition that emits a warning is the condition that raises. A list
of strings makes them two code paths, and two paths encoding the same policy
drift. This is not hypothetical: it is the same failure OS-2 designed against when
its validation section forbade verifying the streaming contract by substring
matching.

Second argument: `warnings` is a **published wire field**. `list[str]` →
`list[object]` later is a breaking change for every client that reads
`warnings[0]` as text. Now is the only cheap moment.

On the taxonomy itself — `admission` / `capability` / `resolution` answers three
different questions:

- `admission` — which **component** produced it.
- `capability` — **why** it happened.
- `resolution` — **what** happened.

A clamp caused by a missing tokenizer is all three simultaneously, so the same
event can be filed under any of them. A taxonomy that cannot classify its own
central example will drift the moment two people use it.

Pick one axis for `type`, and the axis that ages best is **what it means for the
client**, because that is what a client branches on:

- `substituted` — the server ran something different from what you asked.
- `degraded` — the server could not verify something and proceeded anyway. This
  is precisely DEC-047 §4's fail-open signal, and it is the whole reason §9 C.8
  exists.

Component and affected field become structured attributes, not part of the type.
Minimum viable shape — a model, not a framework:

```python
class ResponseWarning(BaseModel):
    type: Literal["substituted", "degraded"]
    code: str          # stable machine identifier, e.g. "max_tokens_clamped"
    message: str       # human text
    field: str | None  # request field affected, when there is one
```

`code` is what strict branches on and what tests assert. `message` is the only
part free to change.

### C5 — The authoritative plan is not in git

`docs/PHASE-A-EXECUTION-PLAN.md` is untracked. This has been housekeeping until
now; OS-4 makes it load-bearing, because OS-4 is the first unit whose scope is
being actively contested (C1) — and the document both sides would cite has no
commit, no diff, and no history. Commit it before the OpenSpec is authored.

### C6 — No metric discontinuity here, and that is worth stating

OS-2 required DEC-050's "prior figures superseded" treatment. OS-4 has no
equivalent: `warnings`/`resolved` are additive and record nothing retroactively,
and switching `count_prompt_tokens` from `getattr` to a declared method does not
change admission behaviour because the chars/4 fallback stays (plan §9 A.3
authorizes it explicitly). Say so in the proposal so nobody invents a
discontinuity clause by pattern-matching on OS-2.

---

## 3. Required changes before OpenSpec

**R1 — Nothing is emitted after the usage event.** Unconditional, whatever else is
decided. (§0)

**R2 — Resolve the §4 / §9 C.6 conflict explicitly, and close it in both
directions.** This is the user's call, not the spec author's. Two coherent
outcomes:

- **Carrier in stream.** Resolution is a dedicated event in the **prologue**
  position — emitted after `admit()` returns, before the first content event. OS-4
  declares `openspec/specs/platform/spec.md` → "OpenAI-compatible streaming event
  order" as **MODIFIED**, not ADDED. Accept the TTFT consequence and record it:
  TTFT is measured from the first *content* chunk, noted as a constraint handed to
  Phase B3.
- **Carrier deferred.** `resolved`/`warnings` land on `ChatCompletionResponse`
  only, the streaming contract is untouched, and the metadata envelope is designed
  once in Phase B with timings in hand. **§9 C.6 must then be amended to scope to
  the non-streaming body** — otherwise Phase A closes unable to meet its own
  acceptance criteria. Streaming clients needing the truth in the interim have
  `strict: true`, which rejects before the stream opens.

Either is defensible. What is not defensible is authoring the OpenSpec without
choosing, or choosing the first without the spec-delta declaration, or the second
without the C.6 amendment.

**R3 — Define `resolved` as the serialized effective request.** Concept named
*Effective Request*, wire name `resolved` unchanged, derivability rule stated
normatively ("a field appears iff it exists on `ChatCompletionRequest`"), and
process-level tier knobs excluded — a tier cap that caused a clamp appears as the
warning's reason, not as a `resolved` field. Record the replay-by-POST property as
the rationale so Phase C inherits it.

**R4 — Type the warnings.** `list[ResponseWarning]` with a single-axis `type`
(`substituted` | `degraded`) plus `code`/`message`/`field`. Normative requirement:
the warning-emitting predicate and the strict-rejection predicate are the same
code path — strict raises on exactly the set the default warns on, verified by a
test that enumerates both sets rather than sampling them.

**R5 — Commit `docs/PHASE-A-EXECUTION-PLAN.md`, and record the strict invariant as
DEC-052** in the sharp form below.

---

## 4. Optional improvements

**O1 — Split OS-4 only if R2 keeps the carrier in-stream.** The plan's stated seam
(`warnings` + `resolved` as read-only truth; `strict` as new rejection behaviour)
holds. With the carrier deferred, the unit is small enough to stay whole.

**O2 — Give the ADR a `code` registry.** One table mapping warning `code` →
meaning, in the ADR rather than scattered across call sites. Three entries today;
the table is what stops the fourth from being invented ad hoc.

**O3 — Assert the empty case.** A test that an unclamped, capability-complete
request returns `warnings: []` — not absent, not null. Truthfulness contracts fail
silently in the direction of never firing, and only the empty-case test catches a
`warnings` array that has quietly stopped being populated.

---

## 5. Strict semantics as a permanent invariant

**Yes — promote it, but sharpen the wording first.**

"Strict changes response policy, never runtime policy" is directionally right and
too loose to enforce: strict changes *admission outcome* (400 instead of a clamp),
which is more than the response body — it is whether the request runs at all. The
testable form:

> **`strict` may only convert a substitution into a rejection. It may never change
> the substitution itself.** Strict partitions outcomes into
> `{rejected, executed exactly as asked}`. It never produces
> `{executed differently}`.

Acceptance criterion, checkable in CI: **for any request accepted under both
modes, the output is byte-identical given an identical seed.**

Make it permanent, in `docs/DECISIONS.md` as DEC-052 rather than as an OS-4
footnote. The pressure to violate it is predictable and comes from a legitimate
place: Phase B and C will want `strict` to also mean "do not batch me," "fail if
the prefix cache is cold," "reject if the model revision differs." Each is
reasonable and each is runtime policy. The invariant pre-answers all of them —
those get their own fields, and `strict` stays one thing. That is the entire value
of writing it down now, while `strict` has exactly one meaning and the cost of
holding the line is zero.

Companion clause, without which the invariant is decorative: **one predicate, two
outcomes** (R4). If the strict rejection set and the default warning set are
computed separately, strict silently becomes its own policy and the invariant
cannot be checked.

---

## 6. Final verdict

**APPROVE WITH MODIFICATIONS.**

The milestone's shape is right: one unit, one concern, degradation made observable
without being tightened, capability hygiene folded in where the path is already
being edited. The abstraction instinct is right too — a dedicated event rather
than an overloaded terminal one, and a `resolved` block rather than scattered
per-field echoes.

Three things must change before the OpenSpec is authored: the event must not sit
after usage (§0), the plan's own §4/§9 conflict must be resolved and closed in
both directions (R2), and the two published wire contracts — `resolved` and
`warnings` — need their invariants fixed now (R3, R4), because both are cheap
today and breaking changes the moment a client reads them.

Next free ADR number: **DEC-052**.
