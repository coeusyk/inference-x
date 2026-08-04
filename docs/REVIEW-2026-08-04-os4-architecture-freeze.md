# OS-4 — Final Architecture Tightening (consistency pass)

Scope: protocol invariants only. No redesign, no implementation change, no scope change.
Reviewed: Effective Request, `resolved` derivability rule, `ResponseWarning`, DEC-052,
streaming lifecycle.

**Assumption this pass builds on:** R2 from the prior review is resolved as
**carrier-in-stream**. Specifying a pre-generation phase presumes there is one.
If the carrier is deferred instead, DEC-053 and both spec requirements below are
deferred with it, §9 C.6 is amended to scope to the non-streaming body, and only
§1.5's streaming-only clause survives.

---

## 1. Required wording changes

### 1.1 "One prologue event" vs "pre-generation metadata lifecycle" — these are not alternatives

They sit on different axes:

- **"One prologue event"** is a *cardinality* claim. It constrains how many events precede content and says nothing about what may go in them.
- **"Fully determined before the first token"** is a *determinacy* claim. It constrains what may go in an event and says nothing about how many there are.

The strong protocol takes both, at different levels of permanence:

> **The determinacy rule is the invariant. The cardinality is the current instantiation.**

Adopt the lifecycle *framing* — it yields a genuinely stronger invariant, because it governs classification rather than counting, and it is what Phase B and Phase C will actually cite. But do **not** adopt "one or more" as OS-4's normative wire text. Doing so has a concrete, verifiable cost: it converts OS-2's exact-sequence assertion — the test its validation section explicitly protected from weakening ("assert order and event count, not merely that certain strings appear") — into a fuzzy test, and it does so for a second event that does not exist. That is YAGNI applied to the wire, and the wire is the expensive place to be wrong.

**Recommended structure:** the phase is named and governed by the invariant; the count is pinned at one and moves only by spec change.

### 1.2 The proposed invariant wording is near-tautological as written

> "Every metadata event emitted before generation must be completely determined before the first token is sampled."

Three defects:

1. **It forbids nothing.** An event emitted before generation is trivially determined before then — you cannot emit undetermined data. The bite has to come from the converse.
2. **"before generation" has no sharp boundary.** When `generate_stream` is called? When the engine schedules? The codebase has exactly one sharp line: `admit()` returning and the effective request being frozen.
3. **"determined" is weaker than needed.** The failure mode is determined-then-revised. The rule needs immutability.

**Defect 2 is the load-bearing one, and it leaks a real Phase B payload.** Anchoring on "before the first token is sampled" admits **queue/wait time** into the pre-generation phase — queue wait ends when the engine begins processing, which is strictly before the first token is sampled. But the prologue is already on the wire by then, so the invariant would classify as prologue-eligible a fact that is physically un-emittable in that slot. The predictable consequence is a Phase B argument about delaying the prologue until after queueing, which would destroy the determined-at-admission property of everything else in it. Re-anchoring closes the leak.

### 1.3 Exact recommended wording

Normative, for `openspec/specs/platform/spec.md`:

> **Requirement: Pre-generation metadata lifecycle**
>
> A streamed response has two metadata phases. The **pre-generation phase** ends
> when the first token is sampled.
>
> An event MAY be emitted in the pre-generation phase **if and only if** every
> field it carries is fully determined and immutable at the moment the effective
> request is finalized. A fact that can change during or after generation MUST
> NOT be emitted in this phase.
>
> The pre-generation phase currently contains **exactly one** event. Adding a
> second is a modification of this requirement, not an extension of it.

And the symmetric clause, without which only half the protocol is defined and Phase B invents the other half:

> **Requirement: Post-generation metadata lifecycle**
>
> Events emitted after the terminal event carry facts about what actually
> happened. The usage event is the last event before `data: [DONE]`. Any future
> post-generation field extends the usage event or is emitted before it. No event
> is emitted after the usage event.

**These are two ADDED requirements, and they do not stand alone.** Carrier-in-stream means the existing requirement `openspec/specs/platform/spec.md` → "OpenAI-compatible streaming event order" is **MODIFIED** by OS-4, because the event sequence it pins now has an event before the first content event. Filing the two requirements above as ADDED while leaving the ordering requirement untouched makes the spec history lie about when the order changed. All three move in the same delta.

Two properties worth noting about this wording:

- The **"if and only if" is what makes it a classification rule** rather than a hygiene note. It routes in both directions, so no future metadata class lands in a phase by default — each one has to answer the question.
- **The anchor point and the principal payload are the same object.** `resolved` *is* the serialized effective request, so it is determined exactly at the anchor by construction. The rule is self-enforcing for OS-4's own payload rather than being a convention someone must remember.

### 1.4 DEC-052 consistency

The strict invariant and the lifecycle invariant are unrelated subjects, so they cannot share an ADR without making DEC-052 the junk drawer that C3 forbade for `resolved`. Since DEC-052 is already spoken for by strict, **the lifecycle invariant is DEC-053.**

Keep the same split both already use: the normative, testable rule lives in `openspec/specs/platform/spec.md`; the rationale and the pre-answered future pressures live in the ADR. DEC-052 records *why* strict may only convert a substitution into a rejection; DEC-053 records *why* metadata is phase-classified. Neither ADR carries the enforcement — the acceptance criteria do.

### 1.5 `ResponseWarning` — one clause to add as a non-goal

Every OS-4 warning originates in admission or tier resolution, so all of them are pre-generation. But `ResponseWarning` is now a type that could in principle be emitted in either phase — a `degraded` warning about a fallback discovered *during* generation is a coherent future case and is barred from the prologue by the invariant.

Add as a non-goal, not as design: **in OS-4 all warnings are pre-generation; a post-generation warning, if one ever exists, attaches to the usage event rather than being retrofitted into the prologue.** One sentence, and it prevents the future mess where the prologue is delayed to accommodate a warning that isn't ready yet.

Also state explicitly that **the lifecycle invariant is streaming-only.** `ChatCompletionResponse` has no phases; it carries `resolved` and `warnings` in one body regardless.

---

## 2. Lifecycle invariant review

**Should it be normative? Yes — in the form given in 1.3, not as proposed.**

The value is not documentation. It is that the invariant *pre-answers* the placement argument for every metadata class that Phase B, C, and D will bring, and it does so with a test each proposal can be run against in one line: *is this fact determined and immutable when the effective request is finalized?* That question has an unambiguous answer for every class enumerated in §3, which is the property a good invariant has and a style guideline does not.

Three things make it enforceable rather than aspirational:

1. **Bidirectionality** (`if and only if`) — it excludes as well as admits.
2. **A physical anchor** — "effective request finalized" is a line in `chat_service.py`, not a phase of the moon.
3. **Immutability, not just determinacy** — closes the determined-then-revised path.

**What it costs:** one real constraint handed to Phase B3 — TTFT must be measured from the first *content* chunk, not from first-chunk-received. That was already true the moment a prologue was accepted; the invariant makes it explicit rather than discovered. Record it in DEC-053's consequences so B3 inherits it instead of finding it.

**What it does not do:** it does not constrain the non-streaming body, does not constrain the error path (DEC-049 already fixes that: error event, then `[DONE]`, no terminal and no usage), and does not license adding events. Cardinality stays pinned.

---

## 3. Future extensibility review

### Phase B timings — belong after generation ✅ (once re-anchored)

| Fact | Determined at effective-request finalization? | Phase |
|---|---|---|
| TTFT | No — not known until the first token exists | post |
| Decode throughput | No | post |
| **Queue / wait time** | **No — determined at dispatch, after the anchor** | **post** |
| Total latency | No | post |

Every Phase B3 timing lands post-generation and they land there *together*. Under the proposed "before the first token is sampled" wording, queue time would have split off from its siblings (see 1.2). The re-anchored wording keeps the class intact, which is the outcome that matters: a metadata class that splits across phases is one that will be relitigated.

### Phase C replay — derives from Effective Request ✅, with one boundary to name

Replay inputs are request-shaped, so the derivability rule admits them: seed, sampling params, effective `max_tokens`. All are determined at the anchor, all are in `resolved`, all are in the prologue. Replay is `POST /v1/chat/completions` with `resolved` as the body. Holds.

**The boundary to name explicitly:** model revision / weights identity is *not* a `ChatCompletionRequest` field, so the derivability rule excludes it from `resolved` — yet it is determined at the anchor (the routed engine is known), so it is prologue-eligible. The two rules are therefore **independent axes**, and the prologue is the superset:

> derivability governs what is in `resolved`; determinacy governs what is in the prologue. `resolved ⊂ prologue`.

State that in DEC-053. Without it, a Phase C unit will reason "prologue means `resolved`" and widen `resolved` to fit a replay handle — reintroducing exactly the junk-drawer failure the derivability rule was written to prevent. OS-4 need not decide where model revision goes; it needs only to prevent the wrong answer being forced.

### No metadata class forced into the wrong lifecycle ✅ — one to pin

| Class | Determined at anchor | Phase | Forced? |
|---|---|---|---|
| `resolved` / effective request | yes, by construction | pre | no |
| Admission + capability warnings | yes | pre | no |
| Deprecation warnings | yes — a property of the request | pre | no |
| Routing decision | yes | pre | no |
| Model revision / weights identity | yes | pre (not in `resolved`) | no — see boundary above |
| Usage | no | post | no |
| Timings (all of B3) | no | post | no |
| Prefix-cache / batching status | no — decided at scheduling | post | no |
| Moderation / content-filter result | no — depends on generated text | post | no |
| Finish reason | no | post (terminal) | no |
| **Trace / request id** | **either — depends on mint point** | **ambiguous** | **pin it** |

Only one genuine ambiguity. A correlation id can be minted at admission (pre) or at completion (post), and the invariant permits both — which means it decides nothing until the mint point is fixed. **Recommendation: fix the mint point at admission.** It is then pre-generation, immutable, and — the reason this is the better answer rather than the arbitrary one — available on the *error path*, where DEC-049 emits an error event and `[DONE]` with no terminal and no usage. A completion-minted id is absent from precisely the responses that most need correlating. One line in DEC-053's consequences; no OS-4 implementation.

---

## 4. Final architecture verdict

The architecture is consistent. Five changes, all wording, none touching scope or implementation:

1. **Adopt the lifecycle framing; keep cardinality pinned at exactly one** (1.1, 1.3). The invariant governs classification; the count moves only by spec change.
2. **Re-anchor the invariant** from "before the first token is sampled" to "at the moment the effective request is finalized," add `if and only if`, add immutability (1.3). As proposed, the wording forbids nothing and leaks queue time.
3. **Add the symmetric post-generation clause** (1.3), which makes §0's no-event-after-usage rule permanent instead of an OS-4 note.
4. **Record the lifecycle invariant as DEC-053**, separate from DEC-052 (1.4), carrying three consequences: TTFT measured from first content chunk; `resolved ⊂ prologue`; trace id minted at admission.
5. **Two non-goal sentences on `ResponseWarning`** (1.5): all OS-4 warnings are pre-generation, a future post-generation warning attaches to the usage event; the invariant is streaming-only.

The three verifications hold. Phase B timings land post-generation as one intact class. Phase C replay derives from Effective Request by `POST`, with the `resolved ⊂ prologue` boundary preventing the one wrong turn available to it. Of the classes enumerated above, exactly one — the trace id — is ambiguous, and it is ambiguous by an unfixed mint point rather than by a defect in the rule.

Effective Request, the derivability rule, `ResponseWarning`, DEC-052, and the streaming lifecycle are mutually consistent, and each of the four future-facing pressures I can name is answered by a rule already written rather than by a decision deferred.

**OS-4 ARCHITECTURE FROZEN**
