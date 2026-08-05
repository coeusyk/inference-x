# Phase A — Execution Plan

- Date: 2026-08-04
- Status: active
- Lifecycle: proposed → active → completed → historical (see §12)
- Authority: subordinate to `docs/DECISIONS.md` (DEC-047), `docs/ARCHITECTURE.md`,
  `CONTRIBUTING.md`, `AGENTS.md`, and `docs/UNDERSTANDING-INFERENCE-X.md` for
  present-tense code description. Where this document appears to conflict with
  any of those, those win and this document is wrong. Architecture review
  documents under `docs/` are historical artifacts, not policy.
- Scope: how to execute Phase A. Not what Inference-X should become. The Engine
  Boundary is settled by DEC-047 and is not reopened here.

---

## 1. Executive summary

Phase A is not housekeeping. It is the milestone that makes every number the
system reports true, and it is a precondition for the credibility of everything
built after it. The strategy is:

**Land the enforcement gate first, then fix the metric that everything else is
measured against, then widen the response contract on top of a correct metric —
and refuse everything else.**

Three properties drive the sequencing:

1. **CI is a hard prerequisite, not a parallel task.** There are 430 tests across
   37 files and no `.github/workflows/` at all. Every subsequent change in this
   milestone edits a hot path (`services/chat_service.py`, `routing/admission.py`,
   `benchmarks/runner.py`) with no automated proof that the other 429 tests still
   pass. Shipping any of Phase A before CI means every later change is
   unverifiable at review time.

2. **Truthful token accounting forces the first real Engine Boundary change, and
   that change must be made deliberately under DEC-047.** See §1.1.

3. **Everything else in Phase A hangs off the corrected metric.** The `resolved` /
   `warnings` block, the advisor weights, and the benchmark suite hash are all
   downstream of "is the number right." Building provenance on a metric that
   counts whitespace-delimited words as tokens records the wrong number,
   precisely and verifiably.

The plan decomposes Phase A into **six OpenSpec changes and one non-spec
housekeeping PR**, with exactly two hard sequencing edges and one shared-file
contention point. Estimated duration is one to one-and-a-half weeks with two to
three contributors, longer if the engine-contract decision is relitigated rather
than decided once at the start of OS-2.

**The dominant risk is not technical.** It is that Phase A gets abandoned partway
because AsyncLLM (Phase B) and the manifest (Phase C) are more interesting. The
plan is structured so that the exit criterion cannot be satisfied by partial
completion, and so that abandonment at any point leaves a coherent repository
rather than a half-migrated one.

### 1.1 Engine Boundary consequence (DEC-047)

Phase A's only change to the Engine Boundary is widening
`BaseEngine.generate_stream`.

Today the method yields bare `str` chunks. A string cannot carry
`usage` or `finish_reason`, so real completion-token counts cannot reach the SSE
terminal chunk without changing that signature. OS-2 therefore adds a streaming
chunk model in `schemas/chat.py` and updates `generate_stream` to yield it.

This conforms to DEC-047:

- The chunk type stays in `schemas.chat` — the package DEC-047 already describes
  as the boundary's current vocabulary.
- It does **not** introduce `inference_x/execution/`, a second type system, or a
  wire↔execution translation layer.
- It is not the start of a backend-neutral execution contract. Backend-neutral
  abstractions remain forbidden until justified by a second concrete
  implementation (DEC-047 Backend Abstraction Principle).

The factory in `engines/registry.py` is **not** in Phase A (§2.4). Declaring
`count_prompt_tokens` on `BaseEngine` (OS-4) is durable capability hygiene from
DEC-047, not a new abstraction layer. Record the streaming widening in the OS-2
ADR; do not reopen DEC-047.

---

## 2. Validate Phase A

### 2.1 Verdict

**Phase A remains the highest-ROI milestone, and its A1–A7 task set survives
DEC-047 intact.** It is not superseded, not reordered at the task level, and not
reduced in scope. What changes is its *decomposition into executable units* and
one previously-invisible architectural obligation inside A1.

### 2.2 Why it still holds

- **DEC-047 §5 reaffirms it directly.** "Phase A (truthful metrics / seed / CI)
  remains first for product truth." Nothing in the accepted architecture displaces
  it.
- **The defects Phase A fixes are still live in the tree.** Verified at the time
  of writing:
  - `observability/middleware.py` — `_count_sse_delta_tokens` counts
    whitespace-delimited words as tokens.
  - `benchmarks/runner.py` — the same approximation feeds `tokens_per_sec`.
  - `services/chat_service.py` — SSE chunks carry neither `finish_reason` nor
    `usage`; the stream ends with a bare `data: [DONE]`.
  - `benchmarks/runner.py` — `suite_version` is read from the prompt-suite JSON,
    never computed; no content hash is verified on load.
  - `schemas/chat.py` — no `seed`, no `stream_options`, no `warnings`, no
    `resolved`.
  - `benchmarks/advisor.py` — `quant_score` is a constant (`1.0`) still weighted
    in the scoring formula.
  - No `.github/workflows/`; `pyproject.toml` dev group contains neither `ruff`
    nor `mypy`.
- **The downstream consumer is blocked on precisely these items.** Varex sends
  `seed` and Inference-X silently drops it. That is worse than not supporting it,
  because the client cannot detect the failure.

### 2.3 What DEC-047 changes about Phase A

DEC-047 does not add work to Phase A. It adds **two constraints and one
obligation**:

1. **Constraint — the streaming contract widening must stay inside `schemas/`.**
   DEC-047 explicitly defers `inference_x/execution/` and dual wire↔execution
   DTOs. A1 cannot be implemented by inventing a backend-neutral chunk type. See
   §3.4.
2. **Constraint — A5 must not silently tighten admission to fail-closed.**
   DEC-047 §4 preserves fail-open-when-unavailable and requires the degradation
   be *typed and observable*. A5's `warnings[]` array is the natural sink for
   exactly that signal, which is why the two belong in one change (§4, OS-4).
3. **Obligation — the durable capability declaration.** DEC-047 names
   `count_prompt_tokens` as durable and permits declaring it on `BaseEngine`.
   `routing/admission.py` currently discovers it via `getattr` with a silent
   chars/4 fallback. Since OS-4 is already editing that discovery path to make
   degradation observable, declaring the capability there costs one method and
   prevents a second edit to the same path later.

### 2.4 What is deliberately *not* pulled into Phase A

**The `engines/registry` factory.** DEC-047 authorises it as hygiene. This plan
defers it, on a rework argument:

The `VLLMEngine` construction path in `api/deps.py` passes several keyword
arguments (`pool_size`, `pool_models`, `pool_configs`, `engine_index`) that exist
only to coordinate multiple engines sharing one GPU. Phase B re-signs that
constructor — AsyncLLM changes engine construction to an async path, and B6
splits multi-model into separate processes, which deletes the pool-coordination
arguments outright. A factory built against today's signature is a guaranteed
rewrite, and it would be a rewrite of the composition root during the highest-risk
migration in the roadmap.

DEC-047 §5 explicitly permits this: factory work is "additive, parallelizable
hygiene (or foldable into later D5) — never a blocking program milestone before
AsyncLLM."

**This deferral creates a documented governance gap that must be recorded, not
left silent.** `docs/ARCHITECTURE.md` states "App construction should go through
`engines/registry` (`create_engine`); route handlers must not import concrete
engines," and `engines/registry.py` is 0 bytes.

Exactly half of that sentence is true today, and the halves have been verified
separately:

- **True.** No handler under `api/routes/` imports a concrete engine. The only
  engine import there is `EnginePool` in `routes/metrics.py` — a container over
  the abstraction, not a backend.
- **False.** The sole `VLLMEngine` import in `api/` is in `deps.py`, where the
  composition root constructs engines directly. That is precisely the site
  ARCHITECTURE.md says should route through `create_engine`.

So the deferral leaves the *handler* half of the guarantee intact and the
*construction* half unmet. That distinction matters: acceptance criterion E.15 and
the standing guarantee in §8.3 assert only the handler half, which Phase A
preserves without doing any work. Handling of the construction half: see §5
(Repository touch points → governance) and §9 (Acceptance criteria). It is tracked
as a named Phase A/B seam item with an owner; the ARCHITECTURE.md text is **not**
weakened to match the code.

### 2.5 The one refinement

The review's "Immediate Next Sprint" proposes a single OpenSpec change,
`2026-08-04-truthful-metrics-and-seed`, covering tasks 1–7. **This plan replaces
that with six changes plus one housekeeping PR.**

Rationale, in order of weight:

1. **`CONTRIBUTING.md` requires one concern per pull request.** A single change
   spanning CI configuration, an engine-contract widening, a sampling parameter,
   a response-schema expansion, a benchmark hash, and an advisor rescoring is six
   concerns. It is also unreviewable — the engine-contract decision would be
   buried among lint config.
2. **The units have genuinely different risk profiles.** CI is zero-risk and
   unblocks everything. The contract widening is the only item that can break the
   OpenAI-compatible surface. The advisor rescoring changes published
   recommendations. Bundling them means the riskiest item gates the safest.
3. **Partial completion must leave a coherent repository.** A single change either
   lands or does not. Six changes give six clean stopping points (§8).
4. **Two of the six can run genuinely in parallel**, which a single change forbids.

Every one of the six traces back to a review task, so the refinement is auditable
rather than a fresh invention:

| Unit | Review task | Concern |
|------|-------------|---------|
| OS-1 | A4 | Enforcement gate |
| OS-2 | A1 | Truthful token accounting (+ engine contract) |
| OS-3 | A3 | Deterministic sampling input |
| OS-4 | A5 + DEC-047 §3/§4 | Truthful resolution + typed degradation |
| OS-5 | A2 | Benchmark suite integrity |
| OS-6 | A6 | Advisor honesty |
| HK-1 | A7 | Repository housekeeping (no OpenSpec change) |

---

## 3. Milestone definition

### 3.1 Objective

**Every number Inference-X reports is either correct or explicitly labelled as
degraded, and a downstream SPRT harness can drive the server with a pinned seed
and detect when the server changed its request.**

Stated as a falsifiable property: after Phase A, for any completed request, a
client can determine (a) the true completion-token count, (b) the sampling
parameters actually used, and (c) every place where the server substituted its
own value for the client's — without inspecting server logs.

### 3.2 Scope

1. **CI enforcement.** GitHub Actions running `uv sync`, `pytest tests/unit`,
   `ruff check`, `mypy src/`. Tool configuration and dev dependencies added to
   `pyproject.toml`. A mypy baseline with `ignore_errors` on the vLLM-touching
   modules is acceptable and expected; a clean mypy run is explicitly not a Phase
   A requirement.
2. **Truthful token accounting.** `stream_options.include_usage` support; a
   terminal SSE chunk carrying `usage`; `finish_reason` on the last content
   chunk; deletion of both whitespace-approximation sites; the engine-contract
   widening that makes this possible.
3. **Deterministic sampling input.** `seed` accepted on the request and threaded
   into vLLM `SamplingParams`.
4. **Truthful Effective Request.** A `resolved` block (serialized Effective
   Request) and a typed `warnings: list[ResponseWarning]` array on the
   non-streaming response and on the streaming **prologue** event; a `strict`
   flag that may only convert substitution into rejection (DEC-052); the
   effective seed echoed in `resolved`; typed and observable admission
   degradation per DEC-047 §4; `count_prompt_tokens` declared on `BaseEngine`.
5. **Benchmark suite integrity.** `suite_version` computed as a hash over the
   canonical prompt list at load, with a loud failure on mismatch, and a
   regeneration path.
6. **Advisor honesty.** Constant `quant_score` removed and weights renormalised;
   `peak_vram_delta_gb` renamed to a name that means what it measures, with a
   compatibility alias.
7. **Housekeeping.** Archive the two settled OpenSpec changes; correct the stale
   test count and the `gpu_memory_utilization: auto` comment.
8. **Decision records.** New ADRs (DEC-048 onward — DEC-047 is the current
   highest) for: prior token counts and all published throughput figures
   superseded; seed support; **strict may only convert substitution into
   rejection (DEC-052)**; recomputed `suite_version` rendering stored benchmark
   results incomparable. A correction note in `article-final.md`.

### 3.3 Explicitly out of scope

Restating from the review's sprint definition and DEC-047's non-decisions, plus
the deferrals this plan adds:

- **AsyncLLM migration and `EngineDriver` deletion** (Phase B1). Phase A must not
  touch `engines/driver.py`.
- **Prometheus `/metrics` passthrough** (Phase B2).
- **Per-request decomposed timings in the response body** (Phase B3). The
  `resolved` block is not a timings block.
- **Admission control rescope or deletion** (Phase B4). OS-4 makes existing
  behaviour observable; it does not change what admission decides.
- **Queueing instead of 429 for `priority: batch`** (Phase B5). A 429 under load
  during the Phase A smoke test is expected evidence, not a bug to fix here.
- **Multi-model process split** (Phase B6).
- **The run manifest** (Phase C).
- **Every consumer-hardware knob** (Phase D).
- **`inference_x/execution/`, backend-neutral DTOs, Execution Contract freeze**
  (DEC-047 non-decisions).
- **A second backend, optional-extra `vllm`, `engines/backends/vllm/` relocation,
  AST import-boundary enforcement** (DEC-047 non-decisions; DEC-007 stands).
- **The `engines/registry` factory** (deferred by this plan, §2.4).
- **Widening `ModelEntry.engine` beyond `Literal["vllm"]`** — DEC-047 forbids this
  without a second registered backend.
- **Any change to the public OpenAI HTTP surface that is not purely additive.**

### 3.4 Architectural boundaries

Three boundary questions arise in Phase A. All three have a forced answer.

**(a) The streaming chunk type — OS-2's central decision.**

`BaseEngine.generate_stream` yields `str`. Real `usage` and `finish_reason` cannot
cross that boundary. Three options exist:

| Option | DEC-047 conformance |
|---|---|
| Yield a chunk model added to `schemas/chat.py` | **Conformant.** DEC-047's own context describes the current state as "methods accept and return `schemas.chat` wire types." Adding a chunk model to that same package continues the existing pattern and introduces no new package, no second type system, and no wire↔execution translation layer. |
| Yield a tuple of `(text, usage \| None)` | Conformant but worse. It avoids a type at the cost of an untyped positional contract that Phase B3 (decomposed timings) will immediately have to widen again, and mypy cannot usefully check it. |
| Introduce a neutral chunk type outside `schemas/` | **Forbidden.** This is the `inference_x/execution/` prohibition in all but name. |

**Decision: add a chunk model to `schemas/chat.py` and widen
`generate_stream` to yield it.** This is a change to the Engine Boundary surface
and must be recorded as such in the OS-2 ADR — including the acknowledgement that
`BaseEngine`'s docstring claim ("Adding a second engine in a future phase must not
require changes here") is now known to have been optimistic, exactly as DEC-047's
problem statement §1 anticipated.

**(b) Capability declaration — OS-4.** `count_prompt_tokens` moves from `getattr`
discovery to a declared method on `BaseEngine`, per DEC-047 §3 (durable).
`kv_capacity_tokens` stays on `getattr` — DEC-047 §3 marks it **provisional** and
explicitly forbids freezing it as a cross-backend contract, because Phase B4 may
rescope or delete it.

**(c) Degradation reporting — OS-4.** One mechanism, two sinks: a structured log
(DEC-047 §4) and the response `warnings: list[ResponseWarning]` array (A5).
Building these as two independent mechanisms guarantees that the second one
rewrites the first. The warning-emitting predicate and the `strict` rejection
predicate are the same code path (DEC-052).

Unchanged boundaries: dependency direction (API → services → interfaces →
implementations) is not altered by any Phase A change. No route handler gains an
import of a concrete engine. Admission does not learn about vLLM internals.

### 3.5 Expected repository impact

Phase A is a small-diff, high-consequence milestone. Rough shape:

- **Net LOC: roughly flat to slightly negative in `src/`.** OS-2 deletes
  `_count_sse_delta_tokens` (~22 lines) and a counting branch in the benchmark
  runner; OS-6 deletes a dead scoring term. OS-4 is the only net-additive change
  of consequence.
- **Test count grows.** Every unit in this milestone changes observable
  behaviour and needs tests that fail if the behaviour regresses. The baseline is
  **430 tests across 37 files** — collected directly, not quoted — and it will move
  meaningfully. The stale counts scattered through `docs/DECISIONS.md` and
  `docs/PHASES.md` will be wrong again immediately, which is why HK-1 fixes the
  count *and* the habit of asserting it in prose.
- **New top-level directory: `.github/workflows/`.**
- **`docs/DECISIONS.md` grows by four to six entries.**
- **`article-final.md` gains a correction note** and stops being a document that
  quietly contradicts the repository.

### 3.6 Independent shippability

Phase A ships without Phase B and without any Phase A successor. It changes no
existing endpoint's shape destructively, adds only optional request fields and
additive response fields, and leaves the engine architecture exactly as DEC-047
found it apart from one deliberate, recorded widening of the streaming contract.

---

## 4. OpenSpec planning

Six OpenSpec changes plus one housekeeping PR. Naming follows the existing
convention (`openspec/changes/YYYY-MM-DD-<slug>/`).

### OS-1 — CI and static-analysis gate  *(review task A4)*

- **Concern:** the repository can prove its own tests pass.
- **Contains:** GitHub Actions workflow (`uv sync`, `pytest tests/unit`,
  `ruff check`, `mypy src/`); `[tool.ruff]` and `[tool.mypy]` in `pyproject.toml`;
  `ruff` and `mypy` **added to the dev dependency group** (neither is currently
  installed); an explicit mypy `ignore_errors` baseline for vLLM-touching modules.
- **Depends on:** nothing.
- **Blocks:** everything.
- **Note:** this is arguably below the OpenSpec threshold — it changes no
  behavioural spec. It is kept as a change because it lands the first
  merge-blocking gate in the repository's history and that deserves a record. If
  the team disagrees, downgrade it to a PR; the sequencing is unaffected.

### OS-2 — Truthful token accounting  *(review task A1)*

- **Concern:** completion-token counts are real.
- **Contains:** `stream_options.include_usage` on the request; the streaming
  chunk model in `schemas/chat.py`; `BaseEngine.generate_stream` widening; the
  vLLM implementation yielding usage and `finish_reason`; the terminal SSE usage
  chunk emitted before `[DONE]`; deletion of `_count_sse_delta_tokens`;
  `observability/middleware.py`'s SSE recorder repointed to read the usage chunk;
  `benchmarks/runner.py` reading `usage.completion_tokens`; ADR declaring prior
  token counts and all published throughput figures superseded; the
  `article-final.md` correction note.
- **Depends on:** OS-1.
- **Blocks:** OS-4, OS-6.
- **Owns exclusively:** the SSE terminal path in `services/chat_service.py`, the
  streaming section of `observability/middleware.py`, and the token-counting and
  `PromptResult`-construction sites in `benchmarks/runner.py`.
- **Critical note:** deleting `_count_sse_delta_tokens` changes what `/v1/metrics`
  reports. Previously recorded metrics become incomparable with new ones — the
  same class of break as the `suite_version` recompute, and it needs the same
  explicit "prior numbers superseded" treatment in the ADR rather than a silent
  cutover.
- **Compatibility check performed:** both existing SSE consumers tolerate an extra
  terminal chunk. `playground/streaming.py` returns on `data: [DONE]` and skips
  lines that yield no token; `benchmarks/runner.py` skips chunks without
  `delta.content`. The additive chunk is backward compatible with both, which is
  why OS-2 can be additive rather than a coordinated cutover.

### OS-3 — Deterministic sampling input  *(review task A3)*

- **Concern:** `seed` reaches the sampler.
- **Contains:** `seed: Optional[int]` on the request; wiring into
  `_sampling_params` in `engines/vllm_engine.py`; a test asserting identical
  output for identical seed at `temperature=0`, single-request, marked `xfail`
  under concurrency with a reference to Phase C3.
- **Depends on:** OS-1.
- **Blocks:** OS-4 (which owns the response echo).
- **Explicitly excludes:** echoing the effective seed in the response. That field
  lives in the `resolved` block, which OS-4 owns. If OS-3 adds a response surface
  for it, OS-4 rewrites it immediately.
- **Do not overclaim.** This change makes seed *reach the sampler*. It does not
  make Inference-X deterministic — batch-composition nondeterminism is untouched
  and belongs to Phase C3.

### OS-4 — Truthful Effective Request and typed degradation  *(review task A5 + DEC-047 §3/§4)*

- **Concern:** the client can see every substitution the server made.
- **Architectural concept:** **Effective Request** — the post-admission request
  the runtime will execute. Wire field name remains `resolved` (OS-2 R4).
  **Derivability rule (normative):** a field appears in `resolved` **iff** it
  exists on `ChatCompletionRequest`. Process-level tier construction knobs are
  **not** fields of `resolved`; a tier cap that caused a clamp surfaces only as
  the clamp warning's reason.
- **Contains:**
  - `resolved` on `ChatCompletionResponse` (non-stream) and on a dedicated
    streaming **prologue** SSE event (`choices: []`, top-level `resolved` +
    `warnings`) emitted after `admit()` returns and **before** the first content
    event. **Nothing is emitted after the usage event** (OpenAI usage-as-trailer
    clients). Exact success order:
    `prologue (resolved+warnings) → content* → terminal finish_reason →
    usage? → [DONE]`.
  - `warnings: list[ResponseWarning]` with
    `type: Literal["substituted","degraded"]`, stable `code`, human `message`,
    and optional `field` (request field affected). Warning emission and
    `strict` rejection share **one predicate** (DEC-052 companion).
  - Effective seed echoed in `resolved` (request-shaped).
  - `strict: bool = false` — may only convert a substitution into a rejection
    (DEC-052); never changes the substitution itself or other runtime policy.
  - Structured logging when an admission gate is skipped for a missing
    capability; `count_prompt_tokens` declared on `BaseEngine`; admission uses
    the declared method (`kv_capacity_tokens` stays on `getattr`).
- **Streaming contract impact:** OS-4 **MODIFIES** the platform requirement
  "OpenAI-compatible streaming event order" (prologue addition). TTFT for
  Phase B3 is defined as time-to-first-**content** chunk (prologue is not
  content).
- **Depends on:** OS-2 (SSE path) and OS-3 (seed field to echo).
- **Blocks:** nothing.
- **Preserves:** fail-open-when-unavailable (DEC-047 §4). No metric discontinuity
  (unlike OS-2 / DEC-050): additive surfaces only; chars/4 admission fallback
  remains (§9 A.3 / C.8).
- **Largest unit in the milestone.** If it needs splitting, the clean seam is
  `warnings` + `resolved` (read-only truth) as one change, `strict` (new
  rejection behaviour) as a second. Do not split along the DEC-047 hygiene line —
  that reintroduces the double-edit this unit exists to prevent.

### OS-5 — Benchmark suite integrity  *(review task A2)*

- **Concern:** `suite_version` verifies something.
- **Contains:** `suite_version` computed as a content hash over the canonical
  prompt list at load in `_load_suite`; a loud raise on disagreement with the
  stored value; a regeneration command (e.g. `make suite-version`); regenerated
  value in `benchmarks/prompts/standard.json`; ADR noting that pre-existing
  stored results are now correctly marked incomparable.
- **Depends on:** OS-1.
- **Blocks:** nothing.
- **Parallelism caveat:** OS-5 and OS-2 both touch `benchmarks/runner.py`, but
  different concerns — OS-5 owns suite load / hash verification; OS-2 owns
  streaming measurement and `PromptResult` construction. Expect clean merges if
  those ownership lines are respected; otherwise rebase before review (§7).

### OS-6 — Advisor honesty  *(review task A6)*

- **Concern:** the advisor's recommendation reflects measured quantities only.
- **Contains:** deletion of the constant `quant_score` and renormalisation of
  advisor weights; rename of `peak_vram_delta_gb` to `vram_footprint_gib` across
  the benchmark runner, schemas, and advisor, with a compatibility alias.
- **Depends on:** OS-2. The rename touches the same `PromptResult` construction
  path that OS-2 rewrites when it stops estimating tokens. Running these
  concurrently conflicts on that shared construction.
- **Blocks:** nothing.
- **Note:** the rename changes a persisted field name. The compatibility alias is
  not optional — stored benchmark results carry the old key.

### HK-1 — Housekeeping  *(review task A7)*

- **Not an OpenSpec change.** Writing an OpenSpec change to archive OpenSpec
  changes is ceremony without content. This is a PR.
- **Contains:** archive `openspec/changes/2026-07-01-fix-engine-driver-late-submit-race/`
  and `openspec/changes/2026-07-01-resolve-default-model-via-variant-selector/`;
  correct the stale test count in `CONTRIBUTING.md`; correct the
  `gpu_memory_utilization: auto` comment in `config/models.yaml`.
- **Depends on:** nothing.
- **Blocks:** nothing.
- **Run it first, on day one.** It is fifteen minutes of work, it is the only unit
  with zero risk, and archiving the settled changes makes `openspec/changes/`
  legible for the six new ones about to land there.

### 4.1 Dependency summary

| Unit | Depends on | Can start |
|------|-----------|-----------|
| HK-1 | — | immediately |
| OS-1 | — | immediately |
| OS-2 | OS-1 | after OS-1 merges |
| OS-3 | OS-1 | after OS-1 merges, **parallel with OS-2** |
| OS-5 | OS-1 | after OS-1 merges, **parallel with OS-2 and OS-3** |
| OS-4 | OS-2, OS-3 | after both merge |
| OS-6 | OS-2 | after OS-2 merges, **parallel with OS-4** |

---

## 5. Repository touch points

Architectural ownership, not code description.

### 5.1 Required

| Area | Ownership change | Unit |
|---|---|---|
| `.github/workflows/` | New. Owns merge admissibility. First enforcement authority in the repository. | OS-1 |
| `pyproject.toml` | Gains tool configuration and two dev dependencies. Becomes the single source of lint/type policy. | OS-1 |
| `schemas/chat.py` | Owns the wire vocabulary. Gains request fields (`stream_options`, `seed`, `strict`), response fields (`warnings`, `resolved`), and the streaming chunk type. Under DEC-047 this package remains *borrowed* vocabulary, not owned long-term. | OS-2, OS-3, OS-4 |
| `engines/base.py` | **The Engine Boundary surface itself.** Gains a widened `generate_stream` return type and a declared `count_prompt_tokens`. The only DEC-047-governed change in the milestone. Every edit here requires an ADR. | OS-2, OS-4 |
| `engines/vllm_engine.py` | Implements the widened contract; threads `seed` into `SamplingParams`. Owns inference execution only — no policy. | OS-2, OS-3 |
| `services/chat_service.py` | Owns the SSE terminal sequence and response minting. The highest-contention file in the milestone. | OS-2, OS-4 |
| `observability/middleware.py` | Owns request-level metric recording. Loses its private token estimator and becomes a consumer of engine-reported truth rather than a second, wrong, estimator. | OS-2 |
| `routing/admission.py` | Owns admission policy. Gains typed degradation reporting; loses one `getattr` capability probe. Policy semantics unchanged — DEC-047 §4 forbids tightening. | OS-4 |
| `benchmarks/runner.py` | Owns benchmark measurement. Stops estimating tokens; starts verifying suite identity. Two units touch it in disjoint functions. | OS-2, OS-5, OS-6 |
| `benchmarks/advisor.py`, `benchmarks/schemas.py` | Own recommendation scoring and its persisted shape. | OS-6 |
| `benchmarks/prompts/standard.json` | Owns the reproducibility primitive's input. Its stored hash becomes verified rather than asserted. | OS-5 |
| `docs/DECISIONS.md` | Owns the decision record. Next entry is DEC-048. | all |
| `article-final.md` | Published claims. Needs a correction note, not a rewrite. | OS-2 |

### 5.2 Optional

| Area | Why optional | Unit |
|---|---|---|
| `tests/unit/` structure | Phase A adds tests to existing files. An integration tier is a real gap (there is none today) but introducing one is not required to make Phase A's numbers true. | — |
| `Makefile` | Gains `suite-version`. Could equally be a script. | OS-5 |
| `playground/streaming.py` | Verified compatible with an additive terminal chunk. Surfacing `warnings` in the TUI is a product improvement, not a Phase A requirement. | — |
| `README.md` | Documents the new request fields. Can trail the code by one PR. | OS-3, OS-4 |

### 5.3 Future

| Area | Owning phase | Note |
|---|---|---|
| `engines/registry.py` (0 bytes) | Phase A/B seam | Deferred by §2.4. **Must be recorded as a named open item**, because `docs/ARCHITECTURE.md` already asserts construction goes through it. The gap is in the code, not the doc — do not weaken ARCHITECTURE.md to match. |
| `api/deps.py` | Phase B | Composition root. Phase B re-signs engine construction; Phase A must not touch it. |
| `engines/driver.py` (226 LOC) | Phase B1 | Deleted by AsyncLLM. Any Phase A work here is wasted. |
| `observability/exporters.py` | Phase B2 | Prometheus passthrough. |
| `utils/vllm_pool_config.py` (558 LOC) | Phase B6 / D | The pool heuristics the multi-model split removes. |
| `schemas/model.py` | Post-DEC-047 ADR | `engine: Literal["vllm"]` is deliberate. Widening requires a second registered backend. |
| `docs/PHASES.md` | Governance | See §5.4. |

### 5.4 Governance touch points

Two inconsistencies exist in the authority set. Both need a decision during Phase
A; neither is a code change.

1. **`engines/registry.py` gap** (§2.4). Recommended handling: a tracking issue
   naming the Phase A/B seam as the target, referenced from the OS-2 ADR. No
   ARCHITECTURE.md edit.

2. **Two competing phase namespaces.** `docs/PHASES.md` numbers phases 0–12 and
   its status table stops at 8, while `docs/REVIEW-2026-08-03-architecture.md` and
   DEC-047 use letters A–E. Two namespaces for one roadmap is drift waiting to
   happen. Recommended handling, following the review's own suggestion: freeze
   `PHASES.md` as the historical record with a header saying so, treat the letter
   phases as the forward roadmap anchored in DEC-047, and make OpenSpec changes
   the only executable unit. **Do not add a "Phase 13" for Phase A.**

---

## 6. Dependency graph

### 6.1 The graph

```
                  HK-1 ─────────────────────────────────► (independent, day 1)

                  OS-1  CI + lint/type gate
                    │
        ┌───────────┼───────────┬───────────────┐
        │           │           │               │
        ▼           ▼           ▼               │
      OS-2        OS-3        OS-5              │
   token truth    seed     suite hash           │
        │           │           │               │
        ├───────────┘           └──► (terminal) │
        │                                       │
        ├──────────────► OS-6 advisor ──► (terminal)
        │
        ▼
      OS-4  resolved / warnings / strict / typed degradation
        │
        └──► (terminal)
```

Hard edges: `OS-1 → {OS-2, OS-3, OS-5}`, `{OS-2, OS-3} → OS-4`, `OS-2 → OS-6`.

### 6.2 First-principles derivation

**Why OS-1 is a hard prerequisite, not a parallel task.**

The repository has 430 tests and no automatic execution. Every subsequent unit
edits a file that other units also edit. Without CI, a reviewer's only evidence
that a change is safe is the author's claim to have run the suite. With five
units in flight across a shared file set, that claim becomes unverifiable in
exactly the situation where verification matters most. CI first is not a
preference; it is what makes the rest of the parallelism in this graph safe.

There is a second reason specific to this milestone: OS-2 changes the shape of
data flowing through `observability/middleware.py`, a file whose failure mode is
silent — a wrong metric does not raise. The tests are the only detector, and a
detector nobody runs is not a detector.

**Why OS-2 precedes OS-4.**

Two arguments, either sufficient.

*Semantic:* the `resolved` block's purpose is to let a client compare what it
asked for against what it got. If `usage.completion_tokens` is still a whitespace
approximation, `resolved` reports a truthful account of a false measurement. It
would be a provenance system over a wrong number — the precise failure the review
identifies as the trap of skipping to Phase C.

*Mechanical:* both units restructure the SSE generator in
`services/chat_service.py`. OS-2 adds a terminal usage chunk and `finish_reason`
on content events; OS-4 adds the Effective Request **prologue** event
(`resolved` + `warnings`) before content, and the same fields on the
non-streaming response. Concurrent edits to that shared path conflict, and the
resolution requires understanding both changes at once — which is exactly the
review burden that separating them was meant to avoid.

**Why OS-3 precedes OS-4 (weakly).**

OS-4 echoes the effective seed. It cannot echo a field that does not exist. This
edge is thin: if OS-3 slips, OS-4 can ship without the seed echo and add it in a
trivial follow-up. Treat it as a soft edge that costs a small PR to break.

**Why OS-6 follows OS-2.**

The `peak_vram_delta_gb` rename touches the `PromptResult` construction path that
OS-2 rewrites when it stops estimating tokens. This is a pure mechanical edge
with no semantic content — but it is real, and a plan that called OS-6
"parallel, benchmarks-only" would be wrong.

**Why OS-5 is parallel despite sharing a file with OS-2.**

OS-5 owns suite load / hash verification; OS-2 owns streaming measurement and
result construction. Those concerns are separable within `benchmarks/runner.py`.
§7 covers residual shared-file risk.

### 6.3 Challenging alternative orderings

**"Seed first — it is one field and unblocks Varex immediately."**

Superficially attractive: OS-3 is the smallest unit and the loudest external
blocker. Rejected because shipping `seed` before CI means shipping the first
Varex-facing behavioural change with no automated regression proof, and because
`seed` without truthful token counts gives Varex a reproducible experiment that
reports throughput wrong by 25–30%. That is a *worse* outcome than the current
state, because it looks trustworthy. Varex would build a corpus on it.

Note this does not delay Varex much: OS-3 runs parallel with OS-2, so it lands
roughly when OS-2 does regardless.

**"Do the `engines/registry` factory first — it is pure hygiene, zero risk."**

Rejected on two grounds. First, the rework argument (§2.4): Phase B re-signs the
constructor the factory would wrap, so the factory is written twice. Second,
DEC-047 §5 explicitly prohibits treating Engine Boundary hygiene as a blocking
milestone before AsyncLLM — putting it at the head of Phase A does exactly that,
even if the intent is benign.

**"Bundle everything into one change, as the review's sprint proposes."**

Rejected in §2.5. The decisive argument is `CONTRIBUTING.md`'s one-concern rule
combined with the fact that OS-2 contains a DEC-047-governed Engine Boundary
decision. That decision needs its own ADR and its own review. Burying it inside a
change that also configures ruff is how architectural drift happens — not through
bad decisions, but through decisions nobody noticed being made.

**"Do OS-4 before OS-2 — client-visible truth is more urgent than metric truth."**

Rejected: it produces an honest report of a dishonest number, and it forces the
`services/chat_service.py` conflict to be resolved in the harder direction (the
larger change rebasing onto the smaller one's restructuring).

**"Skip Phase A; go straight to AsyncLLM, which fixes throughput anyway."**

This is the failure mode the milestone exists to prevent, and it is worth naming
explicitly because it will be proposed. AsyncLLM makes the system faster. It does
not make the reported numbers true — the whitespace token counter is in
`observability/middleware.py` and `benchmarks/runner.py`, neither of which
AsyncLLM touches. Phase B on top of Phase A's absence produces a faster system
that still cannot say how fast it is.

### 6.4 Why this ordering minimises rewrites

Each unit lands on top of a corrected foundation rather than beside one:

- Nothing is built on the whitespace token count, because OS-2 removes it before
  OS-4 and OS-6 build on token data.
- Nothing is built on an undeclared capability, because OS-4 declares
  `count_prompt_tokens` in the same edit that changes its discovery — one edit to
  `routing/admission.py`, not two.
- Nothing is built on the composition root, because Phase A does not touch
  `api/deps.py` and therefore cannot be invalidated by Phase B's rewrite of it.
- The Engine Boundary is widened exactly once, in one unit, with one ADR.
- `kv_capacity_tokens` is deliberately left alone, because DEC-047 §3 marks it
  provisional and Phase B4 may delete it. Declaring it now would be freezing a
  contract the roadmap intends to break.

---

## 7. Merge strategy

Phase A coordination only. Repository-wide git workflow, DEC numbering
conventions, and standing reviewer rules belong in `CONTRIBUTING.md` (or a future
`ENGINEERING.md` if one is introduced) — not in this plan. This section states
only what Phase A's dependency graph and shared-file set require.

### 7.1 Independent workstreams

After OS-1 merges, three streams run concurrently:

- **Stream 1 (runtime contract):** OS-2 → OS-4. The critical path.
- **Stream 2 (sampling):** OS-3. Touches request schema and sampling wiring in
  `engines/vllm_engine.py`.
- **Stream 3 (benchmarks):** OS-5, then OS-6 once OS-2 merges. Entirely within
  `benchmarks/`.

HK-1 runs before all of them and belongs to whoever starts first.

### 7.2 Strict sequencing

- OS-1 before everything.
- OS-2 before OS-4 (hard — shared SSE generator, and semantic).
- OS-2 before OS-6 (hard — shared `PromptResult` construction path).
- OS-3 before OS-4 (soft — costs one trivial follow-up PR to break).

### 7.3 Where conflicts are likely

Ranked by expected pain.

1. **`services/chat_service.py` — the SSE generator.** OS-2 and OS-4 both
   restructure it. *Mitigation:* the hard sequencing edge. Do not parallelise
   these two.

2. **`schemas/chat.py` — three units add fields.** OS-2 (`stream_options`, chunk
   model), OS-3 (`seed`), OS-4 (`strict`, `warnings`, `resolved`). *Assessment:*
   low risk for additive Pydantic fields on distinct concerns. Resolve field
   collisions in review when they occur; do not prescribe editing mechanics here.

3. **`benchmarks/runner.py` — three units touch it.** OS-2 (streaming
   measurement / result construction), OS-5 (suite load / hash), OS-6 (rename).
   *Mitigation:* OS-6's sequencing edge removes the worst pair. For OS-2/OS-5,
   respect the ownership split in §4 and refresh onto `develop` before review if
   both are open.

4. **`docs/DECISIONS.md` — every unit appends an ADR.** Append-only conflicts
   resolve trivially. Claim numbers when opening each unit rather than
   pre-assigning a fixed per-unit block (units need different numbers of
   entries — OS-4 needs at least two).

5. **`engines/base.py` — OS-2 and OS-4 both edit it.** Different methods; units
   already sequenced. Negligible.

### 7.4 Minimising integration risk

- **Land OS-1 before cutting other Phase A branches.** Without CI, later branches
  cannot be verified automatically, and authors will be tempted to fix lint in
  files they do not own.
- **The mypy baseline is written once, in OS-1, and is not expanded in later
  Phase A units.** Baseline growth during OS-2–OS-6 is a review finding (scope
  signal), not a standing repository type-coverage policy.
- **Refresh OS-4 onto `develop` as soon as OS-2 lands**, before OS-4 is
  review-ready. Its conflict surface with the SSE path is known in advance.

---

## 8. Rollback strategy

Phase A is designed so that stopping is always cheaper than reverting.

### 8.1 What survives at each stopping point

| Stopped after | Remains valid | State |
|---|---|---|
| HK-1 | Everything. | Clean. |
| OS-1 | CI, lint and type config. | **Strictly better than the start.** No behaviour changed. Zero revert pressure. |
| OS-2 | Truthful token counts everywhere; widened engine contract; superseded-numbers ADR. | **Coherent and shippable.** This is the highest-value single unit; if only one thing lands, this is the one. |
| OS-3 | `seed` reaches the sampler. | Coherent. Varex's primary blocker is cleared. |
| OS-5 | Verified `suite_version`. | Coherent. Independent of everything else. |
| OS-4 | `resolved` / `warnings` / `strict`; declared `count_prompt_tokens`. | Coherent. |
| OS-6 | Honest advisor scoring. | Milestone complete. |

**No unit leaves the repository in a broken intermediate state.** Every unit is
independently additive at the wire level and independently revertible.

### 8.2 What must be reverted

Almost nothing, but three items have consequences beyond their diff:

1. **OS-2's ADR outlives OS-2's code.** If OS-2 ships and is later reverted, the
   "prior throughput figures superseded" statement and the `article-final.md`
   correction stay true — the old numbers were always wrong. Do not revert the
   ADR with the code. Add a follow-up ADR explaining the reversion.
2. **OS-5's regenerated `standard.json` hash.** Reverting the code without
   reverting the JSON leaves a stored hash nothing computes. Revert both or
   neither.
3. **OS-6's field rename.** The compatibility alias exists precisely so this
   revert is safe. If the alias was omitted, stored benchmark results with the old
   key break on read. Verify the alias landed before considering the rename
   reverted.

### 8.3 Architectural guarantees that hold regardless

These survive any partial completion, because Phase A does not touch what
establishes them:

- **DEC-047's Engine Boundary is intact.** No `inference_x/execution/`, no
  backend-neutral DTOs, no second backend, no optional-`vllm`, no
  `engines/backends/` relocation. The one contract widening stays inside
  `schemas.chat` — the package DEC-047 describes as the boundary's current
  vocabulary.
- **Dependency direction is unchanged.** API → services → interfaces →
  implementations. No handler under `api/routes/` imports a concrete engine at any
  point; the single `VLLMEngine` import in `api/` stays confined to the
  composition root in `deps.py`, which Phase A does not touch.
- **Phase B is unblocked at every stopping point.** Phase A never touches
  `api/deps.py` or `engines/driver.py`. AsyncLLM can start from any point in this
  sequence. This is DEC-047 §5's non-blocking requirement, satisfied structurally
  rather than by promise.
- **The public OpenAI surface remains additive-only.** Every request field is
  optional with a backward-compatible default; every response field is additive.
  An existing client that ignores them behaves identically.

### 8.4 The one irreversible thing

Publishing the correction to `article-final.md` cannot be undone, and should not
be. It is the point at which the repository stops contradicting itself in public.
Sequence it with OS-2 and treat it as a commitment.

---

## 9. Acceptance criteria

Architectural, not implementation. Phase A is complete when all of the following
hold.

**A. Enforcement**
1. A push to `develop` that breaks a unit test, a lint rule, or the type baseline
   fails a required check before a human reads it.
2. The mypy baseline is the one written in OS-1. It has not grown.

**B. Metric truth**
3. No **reported** token count is derived from text. Specifically:
   `usage.completion_tokens`, `/v1/metrics` token figures, and benchmark
   `tokens_generated` all originate from engine accounting, and the whitespace
   approximation exists nowhere in those paths.
   *One authorized exception:* the chars/4 prompt-length heuristic in
   `routing/admission.py`. It is an admission *gate input*, never a reported
   figure, and DEC-047 §4 requires preserving the fail-open fallback it
   implements. Its use is made observable by C.8 rather than removed.
4. For any request, streamed or not, `usage.completion_tokens` originates from the
   engine's own accounting, and the streamed and non-streamed values agree for the
   same prompt and parameters.
5. `/v1/metrics` reports token counts derived from engine-reported usage, and an
   ADR records that figures produced before this change are not comparable to
   figures produced after it.

**C. Contract truth**
6. A client can determine, from the response alone, every parameter the server
   substituted for one the client supplied. For streaming, the response includes
   the prologue Effective Request event (`resolved` + `warnings`) before content;
   for non-streaming, the `ChatCompletionResponse` body. (Carrier-in-stream
   decision closes the former §4 / C.6 ambiguity — see OS-4 unit.)
7. `strict: true` causes rejection where the default causes clamping
   (substitution→rejection only; DEC-052). Both paths are covered by tests that
   enumerate the shared warning/rejection predicate sets rather than sampling
   them.
8. When an admission gate is skipped because an engine capability is unavailable,
   that degradation is recorded in structured logs and surfaced in `warnings`
   as `type: "degraded"` — and admission still fails open, per DEC-047 §4.

**D. Reproducibility primitives**
9. `seed` reaches vLLM's `SamplingParams` and is echoed in the response. No claim
   of end-to-end determinism is made anywhere in the documentation.
10. `suite_version` is computed at load and disagreement with the stored value
    raises. A documented command regenerates it.

**E. Boundary integrity (DEC-047 conformance)**
11. `inference_x/execution/` does not exist. No backend-neutral DTO package
    exists. No second backend implementation exists. `vllm` remains a required
    dependency.
12. `ModelEntry.engine` remains `Literal["vllm"]`.
13. Every change to `engines/base.py` made during Phase A has a corresponding ADR
    stating what was widened and why the widening was unavoidable.
14. `count_prompt_tokens` is a declared method on `BaseEngine`.
    `kv_capacity_tokens` is **not** — it remains provisional per DEC-047 §3.
15. No handler under `api/routes/` imports a concrete engine. (True at the start
    of Phase A — verified — and preserved by it. This criterion asserts the
    handler half of the ARCHITECTURE.md sentence only; the construction half is
    F.16.)

**F. Governance**
16. The `engines/registry.py` gap against `docs/ARCHITECTURE.md` is recorded as a
    tracked open item with a named target seam. `ARCHITECTURE.md` has not been
    weakened to match the code.
17. The phase-namespace question (§5.4) has a recorded decision.
18. `docs/DECISIONS.md` contains entries for: superseded token counts, seed
    support, strict substitution→rejection only (DEC-052), and the
    `suite_version` recompute.
19. `article-final.md` carries the correction note.
20. `openspec/changes/` contains no settled-but-unarchived change.

**G. External validation**
21. One Varex SPRT experiment runs end to end against Inference-X with a pinned
    seed, against `configs/demo_qa_objective.json`, with wall-clock recorded.
22. **A 429 abort under load is a passing outcome**, recorded as Phase B5
    evidence. Phase A is not gated on Phase B5. A plan that requires this run to
    complete cleanly has accidentally made Phase A depend on work it explicitly
    excludes.

---

## 10. Risks

Ranked by expected cost (likelihood × impact).

### Rank 1 — Phase A abandonment for Phase B or C
- **Type:** sequencing / governance
- **Likelihood:** high. Named as the trap in the source review; AsyncLLM and the
  manifest are genuinely more interesting than deleting a word counter.
- **Impact:** high. Building the manifest on approximate token counts produces a
  provenance system that records the wrong number precisely — worse than no
  provenance, because it is trusted.
- **Mitigation:** the exit criteria (§11) are gates, not aspirations. No Phase B
  OpenSpec change is opened before Phase A's exit criteria are met. Acceptance
  criterion G.21 requires external evidence that cannot be produced by
  self-assessment.

### Rank 2 — mypy baseline blocks OS-1 indefinitely
- **Type:** implementation
- **Likelihood:** high. `vllm_engine.py` is 616 lines against an untyped
  third-party API; `vllm_pool_config.py` is 558 lines of numeric heuristics.
- **Impact:** high **if mishandled** — if OS-1 cannot merge, the entire milestone
  stalls at the gate.
- **Mitigation:** an `ignore_errors` baseline is explicitly authorised. A clean
  mypy run is **not** a Phase A goal and must not become one during OS-1 review.
  If OS-1 review starts negotiating type coverage, that is scope creep on the
  critical path; land the baseline and open a separate issue.

### Rank 3 — The engine-contract widening gets relitigated as an Engine Boundary debate
- **Type:** architectural / governance
- **Likelihood:** medium-high. It is a change to `BaseEngine`, one day after
  DEC-047 was accepted, in a repository where the boundary was contested.
- **Impact:** high. Relitigating it either stalls the critical path or produces
  the thick abstraction DEC-047 forbids.
- **Mitigation:** §1.1 and §3.4 decide it once, with the conformance argument
  stated. The OS-2 ADR records the decision *and* the two rejected alternatives.
  DEC-047 is not reopened; this is an application of it. If a reviewer proposes a
  neutral chunk type outside `schemas/`, that is the `inference_x/execution/`
  prohibition and the answer is no.

### Rank 4 — Metric discontinuity handled silently
- **Type:** implementation / credibility
- **Likelihood:** medium. Two separate discontinuities (OS-2's `/v1/metrics`
  cutover, OS-5's `suite_version` recompute) and only one is obvious.
- **Impact:** medium-high. Silently changing what a metric means is the same class
  of dishonesty Phase A exists to fix, committed by the fix.
- **Mitigation:** acceptance criteria B.5 and D.10 require explicit ADRs. Reviewers
  check for them.

### Rank 5 — `services/chat_service.py` conflict resolution loses a change
- **Type:** implementation
- **Likelihood:** medium — near-certain to occur, usually resolved correctly.
- **Impact:** medium. A silently dropped `warnings` population or a lost
  `finish_reason` is invisible until a client depends on it.
- **Mitigation:** hard sequencing (§7.2), early rebase (§7.4), and tests from both
  units surviving the resolution. If OS-4's tests still pass but OS-2's usage
  chunk vanished, CI catches it — which is another reason OS-1 comes first.

### Rank 6 — Overclaiming determinism
- **Type:** governance / credibility
- **Likelihood:** medium. `seed` support is easy to describe as "reproducible."
- **Impact:** medium-high. The project's differentiator is honesty about
  reproducibility. Overclaiming here damages precisely the thing being built.
- **Mitigation:** acceptance criterion D.9. The determinism test is `xfail` under
  concurrency with a Phase C3 reference. Documentation says "seed is honoured,"
  never "runs are reproducible."

### Rank 7 — Scope creep into Phase B via admission
- **Type:** sequencing
- **Likelihood:** medium. OS-4 touches `routing/admission.py`, and once there the
  temptation to fix the 429 behaviour (B5) or rescope the gates (B4) is real.
- **Impact:** medium. Admission rescope before AsyncLLM is work against a
  scheduler that is about to change.
- **Mitigation:** DEC-047 §4 forbids changing fail policy in this decision. OS-4's
  scope is *observability of existing behaviour*, not behaviour. The expected 429
  in the Varex smoke test is evidence to record, not a defect to fix.

### Rank 8 — The `engines/registry` deferral is forgotten
- **Type:** governance
- **Likelihood:** medium-low, but the cost compounds silently.
- **Impact:** low-medium. `ARCHITECTURE.md` continues to describe construction
  that does not happen, and the next reader cannot tell whether it is aspiration
  or drift.
- **Mitigation:** acceptance criterion F.16 requires a tracked item with a named
  seam.

### Rank 9 — Phase-namespace drift
- **Type:** governance
- **Likelihood:** low during Phase A; rises afterwards.
- **Impact:** low individually, corrosive cumulatively.
- **Mitigation:** §5.4 and acceptance criterion F.17.

### Rank 10 — Varex-side blocker prevents the exit test
- **Type:** implementation, cross-repo
- **Likelihood:** low-medium. The test depends on a repository outside this plan's
  control.
- **Impact:** medium. It is the only external validation of the milestone.
- **Mitigation:** the Varex config shape is already known and small. Verify Varex
  can reach a stub endpoint early — during OS-1 or OS-2 — rather than discovering
  a cross-repo blocker at the exit gate.

---

## 11. Exit criteria

Phase A is closed, and planning for the next milestone may begin, when:

1. **All twenty-two acceptance criteria in §9 hold**, verified by inspection of
   the repository rather than by assertion.

2. **All six OpenSpec changes are archived**, and `openspec/changes/` contains no
   unarchived settled change. Per acceptance criterion F.20 this includes the two
   pre-existing ones.

3. **The Varex smoke test has run and its result is recorded** — including a 429
   under load if one occurred, filed as Phase B5 evidence rather than as a defect.

4. **The published-numbers correction is live.** `article-final.md` carries the
   note; the superseding ADR exists. This is the milestone's external commitment
   and it cannot be deferred into the next one.

5. **The two governance gaps are resolved on paper**: the `engines/registry`
   deferral has a named seam and owner, and the phase-namespace question has a
   recorded decision.

6. **CI has been green on `develop` for the full sequence**, not merely on the
   final merge. A milestone whose gate passed only at the end did not have a gate.

### 11.1 What this unblocks, and only that

Meeting these criteria establishes the two preconditions the next milestone
depends on:

- **A trustworthy measurement baseline.** Phase B's central claim is that
  AsyncLLM improves throughput. That claim is only falsifiable against token
  counts that are correct *before* the migration. Without Phase A, Phase B's
  headline result is unmeasurable — the before-number and the after-number would
  be computed differently.
- **A regression detector.** Phase B deletes 226 lines of `EngineDriver` and a
  race class. That is the largest single deletion in the roadmap, and it is only
  safe against a suite that runs automatically.

Nothing beyond these dependencies is decided here. Phase B's internal sequencing,
the admission rescope, and the manifest design are outside this document's scope
and are planned when Phase A closes — not before.

---

## Appendix — traceability

| Review task (REVIEW-2026-08-03 §Phase A) | Execution unit | Status in this plan |
|---|---|---|
| A1 — fix token counting | OS-2 | Expanded: carries the Engine Boundary widening |
| A2 — compute `suite_version` | OS-5 | Unchanged |
| A3 — add `seed` | OS-3 | Narrowed: response echo moved to OS-4 |
| A4 — add CI | OS-1 | Expanded: adds `ruff`/`mypy` as dev dependencies |
| A5 — Effective Request + typed warnings | OS-4 | Expanded: absorbs DEC-047 §3/§4; carrier = streaming prologue |
| A6 — drop `quant_score`, rename VRAM field | OS-6 | Unchanged; sequenced after OS-2 |
| A7 — housekeeping | HK-1 | Unchanged; not an OpenSpec change |
| (review sprint task 6) Varex smoke | Exit criterion 3 | Unchanged |
| (review sprint task 7) ADRs + docs | Distributed across units | Unchanged |

Deferred from Phase A by this plan, with authority:

| Item | Authority | Target |
|---|---|---|
| `engines/registry` factory | DEC-047 §5 (hygiene, non-blocking) | Phase A/B seam |
| `kv_capacity_tokens` as declared capability | DEC-047 §3 (provisional) | Phase B4 |
| Admission fail-closed | DEC-047 §4 (explicitly preserved as fail-open) | Phase B4 |
| Optional-extra `vllm` | DEC-047 §6, DEC-007 | Second-backend slice |
| Second backend | DEC-047 non-decisions | Phase D5, requires future ADR |

---

## 12. Document lifecycle

Execution plans under `docs/` use four states:

| Status | Meaning | Who changes it |
|---|---|---|
| `proposed` | Planning under review; not yet binding for implementation. | Author / reviewers during planning. |
| `active` | Accepted execution guide for an in-flight milestone. | Milestone owner when planning is accepted. |
| `completed` | Exit criteria met; plan remains as the record of what was executed. | Milestone owner when closing the milestone. |
| `historical` | Superseded or no longer operationally useful; kept for audit. | Maintainer, after completion and any follow-up cleanup. |

**Rules:**

- This document starts `active` once the Phase A plan is accepted for execution.
- On Phase A close (exit criteria in §11), set status to `completed`. Do not delete
  the file — OpenSpec archives and ADRs should still be able to cite it.
- Move to `historical` only if a later plan fully replaces it (unusual for a
  completed milestone plan).
- Repository-wide engineering policy discovered while writing a plan belongs in
  `CONTRIBUTING.md`, `AGENTS.md`, or a future `ENGINEERING.md` — not permanently
  in the plan. Plans may *reference* standing policy; they must not *become* it.
