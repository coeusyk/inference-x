# Phase A Final Architecture Audit — Pre-Phase-B

**Context.** Phase A (OS-1 … OS-6) is implemented. This audit verifies that the code
actually satisfies `docs/PHASE-A-ARCHITECTURE.md`, `docs/DECISIONS.md` (DEC-047 …
DEC-057), and `docs/ARCHITECTURE.md` — rather than restating them. Everything below was
checked against source and, where a claim was falsifiable, executed. No files were
modified.

**Gate status (measured, not assumed):** `ruff check .` clean · `mypy src/` clean across
52 files · `pytest tests/unit` **510 passed, 1 xfailed** (the xfail is
`test_seed_determinism.py:85`, correctly citing DEC-051 N1 / Phase C3).

**Label convention.** Findings are prefixed `SEV-` / `OWN-` / `DUP-` / `DOC-` so they can
never be confused with the roadmap's phase labels (B1 = AsyncLLM, C1 = run manifest, …)
from `docs/REVIEW-2026-08-03-architecture.md` §9. Bare `B1`/`C1`/`B4` below always mean
the **roadmap phase**.

---

## Part 1 — Subsystem verification

### 1. Engine Boundary

**Intended (DEC-047).** `BaseEngine` is the sole contract; speaks `schemas.chat` types;
declares `count_prompt_tokens` as durable; leaves `kv_capacity_tokens` provisional and
`getattr`-discovered; construction routes through `engines/registry.create_engine`.

**Actual.** Contract shape is correct. `count_prompt_tokens` is declared on the ABC with
`int | None` and a non-abstract `return None` default — "unavailable" stays representable,
exactly as DEC-047 §4 requires. `kv_capacity_tokens` is a `@property` on `VLLMEngine`
only, undeclared on the ABC — correct. `engines/registry.py` is **0 bytes**;
`api/deps.py:108` constructs `VLLMEngine(...)` directly. Governance exit criterion 2 **is**
satisfied — `CONTRIBUTING.md:39-41` and `AGENTS.md:77,84` both carry the plurality/
anti-`execution/` language.

**Drift.** DEC-047 exit criterion 3 unmet (factory absent). This is *disclosed* in
`PHASE-A-ARCHITECTURE.md` §2 and DEC-047 §5 explicitly declares it nonblocking — so it is
deferred, not drifted. **One real drift**, see **SEV-A2**: the contract's `generate_stream`
declaration is type-unsound and every implementation contradicts it.

**Deferred debt.** Factory + `create_engine`; `kv_capacity_tokens` promotion (owned by
roadmap B4).

**Phase B dependency.** Roadmap B1 (AsyncLLM) implements `BaseEngine` directly. Its
`generate_stream` will be written against the current, defective declaration
(**SEV-A2**). B1 also deletes `engines/driver.py`, which is the only home of
`derive_terminal_metadata` — see **OWN-B5**.

---

### 2. Streaming

**Intended (DEC-049, DEC-050, DEC-053).** `generate_stream` yields `ChatStreamChunk`;
exactly one pre-generation event; usage event last before `[DONE]`; nothing after it; TTFT
from first *content* event; streamed and non-streamed usage derive from one helper.

**Actual.** `ChatStreamChunk` (content / finish_reason / usage) is correct and used
end-to-end. `ChatService.stream_response` emits precisely the documented order.
`observability/middleware.py::_has_content_delta` anchors TTFT on the first content event
with an explicit rationale; `_extract_sse_usage` has **no estimation fallback** and its
docstring forbids reintroducing one — DEC-050 §4 honoured literally. DEC-049 §5's deletion
requirement is **satisfied**: `_count_sse_delta_tokens` has zero occurrences anywhere in
`src/`, `tests/`, `scripts/`, `playground/`. Verified the DEC-053 compatibility claim: the
prologue carries `choices: []`, so `benchmarks/runner.py`
(`(chunk.get("choices") or [{}])[0]`), `middleware._has_content_delta`, and
`playground/app.parse_sse_line` all skip it unmodified, as the ADR asserts.

**Drift.** None in the protocol. Three structural issues: **SEV-A1** (reservation leak at
the prologue `yield`), **OWN-B4** (four independent SSE parsers, no wire-chunk schema),
and **OWN-B5** (the DEC-050 §3 anti-drift helper lives in the module roadmap B1 deletes).

**Deferred debt.** Mid-stream engine failure cannot set `error=True` — documented
Starlette `BaseHTTPMiddleware` limitation, honestly recorded in the module docstring.

**Phase B dependency.** Roadmap B3 adds per-request timings to this stream. Under DEC-053
those are *not* prologue-eligible (queue time is determined after the prologue is on the
wire), so B3 needs a new post-generation slot — and will have to edit all four parsers.

---

### 3. Runtime Resolution

**Intended (DEC-052, DEC-053).** `resolved` + `warnings` on both paths; `resolved` derived
from field membership; `strict` converts substitution → rejection and nothing else; closed
warning-code registry.

**Actual.** Best-implemented subsystem in the repo. `chat_service._resolved()` builds from
`ResolvedRequest.model_fields` reflection over the **effective** request, so membership has
one home and cannot drift. `admission._warn()` emits the response warning and the
structured log in one call — they cannot diverge. DEC-052's "one predicate, two outcomes"
is implemented *textually*: `if request.strict: raise StrictModeViolationError(...)` sits
inside the same branch that emits the `substituted` warning, at both clamp sites
(`admission.py:279`, `:318`). No `strict` check exists on any `degraded` path — correct.
All five registry codes present and no sixth.

**Drift.** None found.

**Deferred debt.** None.

**Phase B dependency.** DEC-052 forbids folding batch-isolation / cold-cache / revision-pin
into `strict`; roadmap B5 (batch queueing) must add its own field.

---

### 4. Deterministic Generation

**Intended (DEC-051 G1–G5).** `seed` accepted, forwarded unchanged, omitted when `None`,
same builder both paths, docs claim "honoured" not "deterministic".

**Actual.** `vllm_engine.py:512-515` — `if request.seed is not None: kwargs["seed"] =
request.seed`. No clamp, no `-1` normalisation (G2/N8 honoured). Single `_sampling_params`
builder used by both `generate` (`:544`) and `generate_stream` (`:574`) — G4 satisfied
structurally, not by convention. `seed` is on `ResolvedRequest`, so it echoes via OS-4.

**Drift.** None found.

**Deferred debt.** Batch-invariance (N1) correctly left as an xfail citing Phase C3.

**Phase B dependency.** None. Roadmap C3 (`deterministic: true` / `VLLM_BATCH_INVARIANT`)
builds on this without rewriting G1–G5, as DEC-051 anticipated.

---

### 5. Benchmarking

**Intended (DEC-054, DEC-055).** One canonicalizer; pinned SHA-256 over parsed prompts;
fail loud naming `make suite-version`; storage filters by suite before latest-per-model;
empty ≠ mismatch.

**Actual.** `suite_identity.py` is the sole hashing implementation and both consumers
(`runner._load_suite`, `scripts/suite_version.py`) call it. `verify_suite` raises
`SuiteIdentityError` naming `make suite-version` for both the missing-key and mismatch
cases. `storage.latest_per_model_for_suite` filters *before* `_latest_per_model` and
returns a `SuiteSelection` NamedTuple whose `status` distinguishes `empty` /
`suite_mismatch` / `ok` — DEC-055 §2 satisfied with a type, not a convention. Both real
consumers (`routes/benchmark.py:55`, `scripts/advise.py:104`) use the suite-aware method.

**Drift.** None in the identity logic. See **DUP-C1**: the non-suite-aware
`latest_per_model()` survives with zero production callers.

**Deferred debt.** `runner._check_vram_budget` compares measured occupancy against an
analytic estimate — DEC-057 §6 recorded this deliberately; leave alone.

**Phase B dependency.** None directly. Under AsyncLLM, `suite_version` becomes visibly
insufficient (it pins input, not runtime) — DEC-055 §4 already says so.

---

### 6. Advisor

**Intended (DEC-056, DEC-057).** No `quant_score`; exact weights 4/9, 1/3, 2/9; `viable`
is the sole viability signal; canonical `vram_device_occupied_gib` with deprecated aliases;
canonical wins on conflict, permanently.

**Actual.** `quant_score` is gone. Weights are literal `(4/9)`, `(1/3)`, `(2/9)` — exact
fractions, not decimals. The module docstring states the weights are uncalibrated editorial
preference and that score is a within-report ordinal — DEC-056 §3/§5 recorded at the code,
not only in the ADR.

DEC-057 §4 (canonical precedence) **verified by execution**:

| Input | Result |
|---|---|
| both `vram_device_occupied_gib=7.0` and `peak_vram_delta_gb=99.0` | → `7.0` (canonical wins) |
| `peak_vram_delta_gb=3.5` alone | → `3.5` (alias reads) |
| `AdvisorResult` with `vram_gb=4.2` | → `4.2`, and `vram_gb` **is** re-emitted on output |

`AliasChoices` lists the canonical name first, which is what makes precedence hold — it is
load-bearing ordering, worth knowing before anyone "tidies" it.

**Drift.** None found.

**Deferred debt.** Both aliases (**DUP-C4**) are dated to Phase A and require an ADR to
remove.

**Phase B dependency.** All five DEC-056/OS-6 advisor assumptions
(`PHASE-A-ARCHITECTURE.md` §10) are invalidated by AsyncLLM — chiefly `_warm_ttft_ms`
dropping `prompt_results[0]`, and a scalar `mean_throughput_tps` weighted at 44 % with no
concurrency qualifier.

---

### 7. Admission

**Intended (DEC-047 §4, DEC-053).** Three gates; capability via the declared method; typed
+ observable degradation; degraded never rejects; caller must always `release()`.

**Actual.** `_prompt_tokens` calls `engine.count_prompt_tokens()` — no `getattr` — and
falls back to chars/4 *with* a `prompt_tokens_estimated` warning. The docstring correctly
scopes that heuristic as "a gate input, never a reported figure". No degraded path raises.

**Drift — confirmed defect.** `admit()`'s own docstring states: *"The caller MUST call
release() with that same reserved_tokens value once the request completes (success or
failure)."* `ChatService.stream_response` violates this on one path. See **SEV-A1** — the
only finding in this audit I was able to reproduce as a live failure.

**Deferred debt.** `kv_capacity_tokens` `getattr` discovery — sanctioned by DEC-047 §3,
but it exists at **two** sites (`admission.py:305`, `routes/metrics.py:43`), see
**OWN-B2**.

**Phase B dependency.** Roadmap B4 rescopes admission to what the vLLM scheduler cannot
do. It will delete or rewrite the KV gate — and inherits **SEV-A1**'s accounting unless
fixed first.

---

### 8. Observability

**Intended (DEC-049, DEC-050, DEC-053).** No estimation ever; absence over wrong numbers;
TTFT from first content event; exactly one record per request.

**Actual.** Fully compliant. The SSE wrapper records `tokens_per_sec=None` when no usage
event arrived rather than 0 or an estimate. Client disconnect (`GeneratorExit`) still
records what was captured. The double-count hazard is handled by returning early from
`dispatch()` for the SSE branch.

**Drift.** Documentation only: the inline comment at `middleware.py:103` still says the
wrapper records *"an approximate token count"* — a leftover from the pre-DEC-050 design
that now contradicts the module docstring twenty lines above it. See **DOC-D1**.

**Deferred debt.** No `Exporter` protocol (**DUP-C3**).

**Phase B dependency.** Roadmap B2 passes vLLM's native stat loggers through to
Prometheus. This middleware measures at the HTTP boundary and will overlap it; DEC-050's
discontinuity rule applies again if the two ever feed one series.

---

### 9. Composition Root

**Intended (`ARCHITECTURE.md`).** `api/deps.py` wires everything; business logic stays out
of route handlers; dependency flow `API → Services → Interfaces → Implementations`.

**Actual.** `deps.py` correctly owns registry / pool / router / admission / recorder as
`lru_cache` singletons with a matching `shutdown_app()`. Fail-open tier resolution is
consistent with the rest of the codebase.

**Drift — three findings.** `api/routes/benchmark.py` bypasses the composition root
entirely and *is* a service (**OWN-B1**). `api/routes/metrics.py` reaches past `BaseEngine`
into a concrete engine attribute (**OWN-B2**). VRAM-tier resolution is written three times
and the available-VRAM fallback twice, with divergent expressions feeding the same
`select_variant` call (**DUP-C2**).

**Deferred debt.** Direct `VLLMEngine` construction pending the DEC-047 factory.

**Phase B dependency.** Roadmap B6 (split multi-model into two processes) rewrites
`_build_engine_pool` and deletes `_POOL_STEP_LOCK`. Roadmap B1 changes what `create_engine`
would construct — which is the argument for *not* building the factory before B1 lands.

---

## Part 2 — Hidden architectural debt

### A. Immediate — correctness or contract defects

**SEV-A1 · Admission reservation leaks on early SSE disconnect** — `Phase B blocker`
`services/chat_service.py:139-145`

The DEC-053 prologue `yield` sits **outside** the `try/…/finally` that calls
`self._admission.release(...)`. Everything from `admit()` to that `yield` is unguarded, so
a `GeneratorExit` delivered at the prologue suspension point never reaches the release.

**What I proved.** Driving `ChatService.stream_response` against a stub engine and calling
`aclose()` after consuming only the prologue:

```
after FULL consumption  : KV reserved 0    | seq in-flight 0     ← correct
after EARLY disconnect  : KV reserved 522  | seq in-flight 1     ← leaked
```

That establishes the defect precisely: **the guard does not cover the prologue suspension
point.**

**What follows in production (inferred, not executed).** In the live app this generator is
wrapped by `middleware._wrap_and_record_sse`, which consumes it with `async for` — and
`async for` does *not* close its iterator. On client disconnect the inner generator is
finalized by the asyncgen GC hook instead, so the release is either lost outright or runs
nondeterministically late. Both outcomes are bugs; I am flagging the propagation path as
inferred rather than measured.

The sequence-slot leak is the serious half: `_seq_tracker` is compared against
`max_num_seqs` and never decays, so **N disconnects permanently cost N concurrency slots**
and the model eventually returns `EngineSaturatedError` (429) forever until restart.
Reachable by ordinary client behaviour — a closed browser tab, `curl` Ctrl-C, an LB read
timeout.

**Provenance.** Before OS-4 the first suspension point after `admit()` was inside the
`try`. DEC-053's prologue **introduced** this by placing a `yield` in front of the guard.
It is not a pre-existing defect that OS-4 merely reshaped.

**SEV-A2 · `BaseEngine.generate_stream` is declared as a coroutine, not an async
generator** — `Phase B prerequisite` · `engines/base.py:39`

Declared `async def generate_stream(...) -> AsyncGenerator[ChatStreamChunk, None]`. Because
the ABC body has no `yield`, mypy reads it as
`Coroutine[Any, Any, AsyncGenerator[ChatStreamChunk, None]]` — i.e. "await me, then iterate".
Every real implementation is a true async generator, so the declared contract is
contradicted on both sides:

- `vllm_engine.py:556` — *"Return type `AsyncGenerator[...]` incompatible with return type
  `Coroutine[..., AsyncGenerator[...]]` in supertype `BaseEngine`"* (invalid override).
- `chat_service.py:155,158,206` — `__anext__` / `aclose` on a `Coroutine` (3 errors).

Both are invisible today because `engines.vllm_engine` and `services.chat_service` are
`ignore_errors = true` in the DEC-048 baseline. This is the single most important boundary
in the system and it is currently type-unsound *and* suppressed.

DEC-048 named this exact defect and DEC-049's migration notes said OS-2 should clear the
`chat_service` entry "when the contract type-checks". OS-2 widened the *payload* (`str` →
`ChatStreamChunk`) but never touched the *declaration*, so the errors survived with only
their type names changed.

**Fix verified in a sandbox copy** (repo untouched): dropping `async` from the ABC
declaration clears **all four** errors — `chat_service` goes to *"Success: no issues
found"*, and `vllm_engine`'s supertype-incompatibility error disappears (9 → 8, the
remainder being vLLM's untyped `LLM`). `base.py` itself still type-checks clean; the
docstring-only abstract body produces no missing-return error.

**Why before Phase B:** roadmap B1 writes an AsyncLLM implementation against this contract.
Ship it as-is and the invalid-override relationship and its suppression are inherited
permanently, at the moment DEC-049 §3 requires the chunk contract be preserved across the
driver's deletion.

**SEV-A3 · Stale suppression: `services.chat_service` in the mypy baseline** —
`Phase B prerequisite` · `pyproject.toml`

Consequence of SEV-A2, listed separately because DEC-048 §4 rules explicitly that
*"over-suppression is a defect, not a safe default"*. The entry is currently masking
exactly one defect — the one DEC-049 promised to remove. Verified `api/deps.py`'s two
baselined errors are also still live (`select_variant` receiving `VramTier | None`;
`MetricsRecorder(exporter=…)` — see DUP-C3). `engines/driver` (8) and `engines/vllm_engine`
(9, mostly vLLM's untyped `LLM`) are legitimately deferred; driver disappears with roadmap
B1.

**SEV-A4 · Phase A is entirely uncommitted; OpenSpec changes unarchived** —
`Phase B prerequisite` · process

`git status` shows all OS-5/OS-6 work as modified/untracked, including new files
(`suite_identity.py`, `test_suite_identity.py`, `suite_version.py`,
`PHASE-A-ARCHITECTURE.md`). Three change dirs sit unarchived at `openspec/changes/`
(`2026-08-04-add-ci-and-static-analysis-gate`, `verify-benchmark-suite-identity`,
`add-honest-advisor-scoring`) with their `tasks.md` verification checkboxes — including
V27/V28 — still unticked, plus two stale `2026-07-01-*` dirs. Starting Phase B on top of an
uncommitted Phase A means the first B1 diff is unreviewable against a stable baseline.

### B. Ownership and dependency-direction violations

**OWN-B1 · `api/routes/benchmark.py` is a service wearing a route's clothes** —
`Technical debt` · `api/routes/benchmark.py:23-24,49-88`

Three compounding violations of `ARCHITECTURE.md`'s "keep business logic out of route
handlers":
1. Module-level `_store = ResultStore()` / `_advisor = ModelAdvisor()` — two components
   constructed at import time, bypassing `api/deps.py` entirely. Every other subsystem
   uses `Depends(...)`.
2. The handler orchestrates the whole use case: suite-version resolution → storage
   selection → status branching → registry load → `model_max_lens` assembly → advisor
   invocation. There is no `BenchmarkService`; the route is it.
3. `ModelRegistry.from_config(get_settings().config_dir)` is called **per request**,
   bypassing `deps._build_registry`'s `lru_cache` — duplicate construction logic plus
   per-request YAML I/O on a read-only endpoint.

Roadmap C1 (`GET /v1/manifest`, `X-Run-Id`) lands adjacent to this and will inherit the
shape.

**OWN-B2 · `kv_capacity_tokens` discovered by `getattr` at two sites, one of them a route**
— `Technical debt` (fix during roadmap B4) · `routing/admission.py:305`,
`api/routes/metrics.py:43`

DEC-047 §3 sanctions `getattr` for this provisional capability, so neither site is a
violation on its own. But `routes/metrics.py` is a **route** reaching past `BaseEngine`
into a concrete engine's attribute, which DEC-047's ownership table forbids for the API
layer, and having the discovery duplicated means roadmap B4 must find and fix both. The
route also imports `estimate_weight_gib` / `probe_gpu_memory_gib` from
`utils/vllm_pool_config` directly — more computation in a handler.

**OWN-B3 · `benchmarks/` imports `services/` and `utils/`** — `Leave alone`
`benchmarks/runner.py:19-20`

`runner.py` pulls `services.model_service.ModelRegistry` and
`utils.vllm_pool_config.estimate_engine_footprint_gib`. Benchmarks are an offline path
orthogonal to the request spine, so this is a layer crossing rather than an inversion, and
DEC-057 §6 already recorded the `_check_vram_budget` coupling as known-and-unfixed. Noted
for completeness; no action.

**OWN-B4 · The SSE wire format has four parsers and no schema** — `Phase B prerequisite`
`observability/middleware.py`, `benchmarks/runner.py`, `playground/app.py`,
`services/chat_service.py`

DEC-053 calls the event order "the protocol", but nothing in `schemas/` represents a wire
chunk. `ChatStreamChunk` is the *engine-facing* type; the OpenAI-facing chunk is
hand-built as dict literals in `chat_service` (and the timeout error event is a raw
f-string with escaped quotes, bypassing the local `_event()` helper). Three consumers then
independently re-derive `data:` stripping, the `[DONE]` sentinel, usage extraction, and
delta-content extraction.

It works today — I verified all four agree on the DEC-053 prologue. The cost is forward:
roadmap B3 adds a timing event to this stream, and every added field is a four-site edit
against four hand-rolled parsers with no shared definition to check against.

**OWN-B5 · The DEC-050 anti-drift helper lives in the module roadmap B1 deletes** —
`Phase B prerequisite` · `engines/driver.py:41`, `engines/vllm_engine.py:13-16,598`

DEC-050 §3 states that streamed and non-streamed usage *"both now derive from the single
`derive_terminal_metadata` helper so they cannot drift apart again"*, and
`PHASE-A-ARCHITECTURE.md` §3 repeats that as a standing Phase A property. That helper is
defined only in `engines/driver.py:41`. `vllm_engine.py` imports it across the module
boundary (`:13-16`) and calls it in the streaming path (`:598`); the driver calls it
internally for the non-streaming path (`:232`).

Roadmap B1 deletes `driver.py`. If B1 reimplements terminal-metadata derivation inside the
AsyncLLM engine rather than relocating the existing function, DEC-050 §3's guarantee is
silently voided — the two paths get two implementations again, which is the exact failure
DEC-050 was written to close. The helper must be **moved**, not rewritten, and B1's change
document should say so explicitly. This is simultaneously a "helper Phase B should delete"
and a "file whose responsibility changed since DEC-047": `driver.py` began as a vLLM
execution wrapper and now also hosts a protocol-level invariant.

### C. Dead abstractions, duplication, and temporary aliases

**DUP-C1 · `ResultStore.latest_per_model()` is dead and is the exact DEC-055 footgun** —
`Technical debt` · `benchmarks/storage.py:73-77`

Zero production callers — `routes/benchmark.py` and `scripts/advise.py` both use
`latest_per_model_for_suite`. Only tests and the internal `_latest_per_model` reference it.
DEC-055 preserved it for compatibility, but what survives is a public method that returns
mixed-suite results — precisely the silent wrong answer DEC-055 was written to prevent. A
future contributor reaching for the shorter name gets the unsafe behaviour with no warning.

**DUP-C2 · VRAM-tier resolution written three times; available-VRAM math written twice** —
`Technical debt` · `api/deps.py:36-52, 92-96, 155-160, 168-179`

`get_settings().get_vram_tier()` wrapped in fail-open try/except appears in
`_resolve_vram_tier_for_pool()`, inline in `_build_admission_controller()`, and inline in
`initialize_app()` — one policy, three implementations.

More consequential: `_build_engine_pool` and `_build_router` each independently probe the
GPU and compute an available-VRAM fallback using **different expressions**
(`total = session_total or 8.0; available = session_free ?? total` vs
`available = session_free ?? (session_total or 8.0)`), then feed the result to the same
`select_variant()`. They agree today. If they ever diverge, the router resolves a family to
a variant the pool did not load, and every request dies on `_resolve_engine`'s "Routed to
model X but loaded model is Y". Two probes of the same device in one startup is also
avoidable.

**DUP-C3 · No `Exporter` protocol** — `Technical debt` · `observability/recorder.py:29`

`MetricsRecorder.__init__(exporter: NullExporter | None = None)` is typed to a *concrete*
class. `build_exporter()` returns `NullExporter | JsonLineExporter`, so `deps.py:206`
passing a `JsonLineExporter` is a genuine type error — currently invisible under the
`api.deps` baseline entry. Two implementations, duck-typed across a module boundary, in a
codebase that types everything else. Same decorative-boundary class DEC-047 §1 named for
the engine; a three-line `Protocol` closes it.

**DUP-C4 · DEC-057 compatibility aliases** — `Leave alone (tracked)`
`benchmarks/schemas.py:43-46, 65-73`

`peak_vram_delta_gb` (read-only) and `vram_gb` (read **and** re-emitted via
`@computed_field`) both behave exactly as DEC-057 §3/§4 specify — verified by execution.
They must eventually disappear, and DEC-057 requires an ADR to remove them. Correct as-is;
listed so the obligation stays visible. Note that `AliasChoices` argument **order** is what
implements the permanent canonical-precedence rule.

### D. Documentation-only

**DOC-D1 · Stale comment contradicting DEC-050** — `Documentation only` ·
`observability/middleware.py:103`

*"wrap body_iterator so it still records TTFT and an approximate token count"* — the
approximation was deleted by OS-2 and DEC-050 §4 forbids reintroducing it. The module
docstring and `_extract_sse_usage`'s docstring both say the opposite, correctly. This is the
one surviving comment that would mislead someone into "restoring" the fallback.

**DOC-D2 · `docs/ARCHITECTURE.md` package path is misspelled** — `Documentation only`

Five occurrences of `src/inferencex/` (no underscore); the package is `src/inference_x/`.
The Benchmark-layer section uses the correct spelling, so the file contradicts itself. The
same section still describes the advisor as *"weighted scoring (throughput, warm TTFT,
VRAM headroom)"* without the DEC-056 measured-only framing — accurate but pre-OS-6 in tone.

---

## Part 3 — Scores

### Phase A Architecture Score — **88 / 100**

Every OS-2…OS-6 decision is implemented as specified, and several are implemented in the
one way that makes them *structurally* unable to drift rather than merely conventionally
correct: `_resolved()` derives membership by reflection, `_warn()` fuses log and response
warning, DEC-052's strict-raise sits textually inside the branch that warns, and a single
canonicalizer owns suite identity. The DEC-054 digest and DEC-057 precedence rule both hold
under execution. Deductions: the Engine Boundary's own `generate_stream` declaration is
type-unsound and suppressed (**SEV-A2**) — a decorative-contract defect of exactly the kind
DEC-047 §1 exists to name; and the route layer contains a full service (**OWN-B1**) plus a
`getattr` past `BaseEngine` (**OWN-B2**). The unbuilt factory is *not* deducted — DEC-047 §5
authorises the deferral and `PHASE-A-ARCHITECTURE.md` §2 discloses it.

### Phase A Maintainability Score — **82 / 100**

510 passing tests, a green three-tool gate, and the best decision record I have audited —
the ADRs record *why* and what was rejected, and the code cites them at the point of use,
so intent survives without archaeology. Against that: four hand-rolled SSE parsers with no
shared wire schema (**OWN-B4**), a protocol-level invariant parked in a module scheduled for
deletion (**OWN-B5**), triplicated tier resolution and duplicated VRAM math with a real
divergence hazard (**DUP-C2**), a dead public method that is also a correctness footgun
(**DUP-C1**), a suppression masking a live contract defect (**SEV-A3**), a missing exporter
protocol (**DUP-C3**), and a 636-line `vllm_engine.py`.

### Phase B Readiness Score — **74 / 100**

The seams are in the right places: the driver is isolated behind `BaseEngine`, admission is
a separate object with a narrow interface, the chunk contract is payload-correct, and
DEC-047 §5 already ruled Engine Boundary hygiene nonblocking. What holds it back is
concrete and small: a reproducible reservation leak that roadmap B4 would otherwise inherit
(**SEV-A1**), a contract declaration that B1's AsyncLLM implementation would inherit as a
permanent invalid override (**SEV-A2**), an anti-drift guarantee whose only home B1 deletes
(**OWN-B5**), and an uncommitted, unarchived Phase A (**SEV-A4**) that leaves B1's first
diff without a stable baseline. None is large. All are cheaper now than after B1 lands.

---

## Part 4 — Prioritized list

### Immediate before Phase B

1. **SEV-A1** — Fix the admission reservation leak. Bring the prologue `yield` inside the
   guard so `release()` covers every path from `admit()` onward. Add a regression test for
   early-disconnect (`aclose()` after the first event) asserting both trackers return to
   zero. *Restores a documented invariant; no new architecture.*
2. **SEV-A2 + SEV-A3** — Correct `BaseEngine.generate_stream` to declare an async-generator
   return (drop `async` from the ABC declaration), then remove `services.chat_service` from
   the DEC-048 mypy baseline. Verified in a sandbox to clear exactly 4 suppressed errors
   (3 in `chat_service`, the invalid override in `vllm_engine`), delivering what DEC-049's
   migration notes committed to. Do this **before** roadmap B1 writes an implementation
   against the contract.
3. **SEV-A4** — Commit Phase A, tick the OS-1/OS-5/OS-6 `tasks.md` verification checklists,
   `openspec archive` all three, and resolve the two stale `2026-07-01-*` change dirs.
4. **OWN-B5 (design constraint only)** — Write into roadmap B1's change document that
   `derive_terminal_metadata` must be **relocated**, not reimplemented, when `driver.py` is
   deleted, citing DEC-050 §3. No code change now; recording it before B1 starts is what
   prevents the guarantee lapsing silently.
5. **DOC-D1** — Delete the stale "approximate token count" comment (one line; it actively
   invites a DEC-050 violation).

### Phase B work

6. **OWN-B4** — Give the OpenAI SSE chunk a typed home in `schemas/` and route the parsers
   through it, as part of roadmap B3 rather than before it — B3 is the change that makes
   four parsers expensive.
7. **OWN-B2** — Collapse `kv_capacity_tokens` discovery to one site during roadmap B4
   (admission rescope), and get it out of `routes/metrics.py`. DEC-047 §3 defers the
   durability decision to exactly this point.
8. **DUP-C3** — Introduce an `Exporter` protocol while roadmap B2 (Prometheus passthrough)
   is adding a third exporter; clears one `api/deps` baseline error.
9. **DUP-C2** — Fold the triplicated tier resolution and duplicated VRAM math into one
   helper while roadmap B6 rewrites `_build_engine_pool`.
10. **DEC-047 factory** — build `create_engine` **after** roadmap B1, not before. B1 changes
    what it would construct; building it first means writing it twice.

### Phase C work

11. **OWN-B1** — Extract a `BenchmarkService` and move `_store`/`_advisor` into `deps.py`
    when roadmap C1 adds `GET /v1/manifest` to the same router.
12. **DUP-C1** — Remove or privatise `ResultStore.latest_per_model()` once no test depends
    on it; the run manifest (roadmap C1) is when comparability keys get revisited anyway.
13. **Advisor assumption review** — after roadmap B1/B3 land, revisit all five
    `PHASE-A-ARCHITECTURE.md` §10 assumptions (`_warm_ttft_ms`, scalar throughput,
    device-occupancy attribution). AsyncLLM invalidates them; DEC-056 §3 already permits
    reweighting.

### Future (ignore for now)

14. **DUP-C4** — DEC-057 alias removal. Requires its own ADR; correct as-is.
15. **DOC-D2** — `docs/ARCHITECTURE.md` path spelling and pre-OS-6 advisor phrasing.
16. **OWN-B3** — `benchmarks/` → `services/`/`utils/` imports. Recorded, not a defect.
17. `vllm_engine.py` size. Only worth touching if roadmap B1 does not already restructure
    it.

---

## Verification performed

- `ruff check .` · `mypy src/` · `pytest tests/unit -q` — all green (510 passed, 1 xfailed).
- Each of the four DEC-048 baseline modules re-checked **without** suppression
  (`mypy --config-file=/dev/null --follow-imports=silent`) to establish which entries are
  still load-bearing: `chat_service` 3, `deps` 2, `driver` 8, `vllm_engine` 9.
- **SEV-A2's proposed fix applied to a sandbox copy of `src/`** (repo untouched) and
  re-checked: `chat_service` → clean, `vllm_engine` 9 → 8 with the supertype-incompatibility
  error gone, `base.py` itself clean. The "clears 4 errors" claim is measured, not assumed.
- DEC-057 canonical precedence exercised directly against `BenchmarkResult` /
  `AdvisorResult` with conflicting and alias-only payloads.
- **SEV-A1** reproduced end-to-end with a stub engine: full consumption releases correctly;
  `aclose()` after the prologue leaks 522 KV tokens and 1 sequence slot. The production
  propagation path through `_wrap_and_record_sse` is inferred from `async for` semantics,
  not executed — stated as such in the finding.
- DEC-049 §5 exit condition checked: `_count_sse_delta_tokens` has **zero** occurrences in
  `src/`, `tests/`, `scripts/`, `playground/` — the helper is genuinely deleted.
- DEC-053 additive-compatibility claim re-checked against all three prologue consumers.
- DEC-050 §3 traced to source: `derive_terminal_metadata` defined at `driver.py:41`,
  imported by `vllm_engine.py:13-16`, called at `vllm_engine.py:598` and `driver.py:232`.
- DEC-047 exit criteria walked individually: 1 ✅, 2 ✅ (`CONTRIBUTING.md:39-41`,
  `AGENTS.md:77,84`), 3 ❌ (deferred, disclosed), 4 ✅, 5 ✅, 6 ✅, 7 ✅.
