# Tasks — OS-4 Truthful runtime resolution and typed degradation

Repository order. Every touched file appears exactly once as a section owner.
Each task carries **why**, **after**, and **check**.

## 0. Preconditions

- [x] 0.1 Confirm OS-1, OS-2 and OS-3 are archived under `openspec/changes/archive/`
      and `openspec/changes/` contains only this change.
      **Why:** OS-2 owns the SSE generator this change extends; OS-3 owns the `seed`
      field `resolved` echoes. **After:** nothing. **Check:** `ls openspec/changes/`.
- [x] 0.2 Confirm DEC-052 is present and `accepted` in `docs/DECISIONS.md`, and that
      DEC-053 is unused.
      **Why:** this change implements DEC-052 and writes DEC-053; a number collision
      is silent and expensive. **After:** nothing.
      **Check:** `grep -n '^### DEC-05' docs/DECISIONS.md`.
- [x] 0.3 Record the baseline: `uv run pytest tests/unit -q` passes, and note the
      pass/xfail counts.
      **Why:** the xfail is OS-3's seed-under-concurrency deferral to Phase C3, not a
      defect; a second xfail appearing during this change is a defect. **After:** 0.1.
      **Check:** counts recorded in the PR description.

## 1. Decision record — `docs/DECISIONS.md`

- [x] 1.1 Add **DEC-053**, status `accepted`: pre-generation and post-generation
      metadata lifecycle.
      **Why:** AGENTS.md requires an accepted ADR for a contract this durable, and the
      spec requirement needs a rationale home. **After:** 0.2. **Check:** the entry
      states (a) the if-and-only-if determinacy rule anchored at effective-request
      finalization, (b) cardinality exactly one, (c) usage is last before `[DONE]`.
- [x] 1.2 Record DEC-053's three consequences explicitly: TTFT is measured from the
      first content event; `resolved ⊂ prologue`; the warning-code registry table.
      **Why:** each is a rule a later phase will otherwise re-derive or contradict.
      **After:** 1.1. **Check:** all three appear under Consequences.
- [x] 1.3 Do not reopen DEC-047, do not rewrite DEC-049, DEC-050, DEC-051 or DEC-052.
      **Why:** all five are accepted and none is contradicted by this change.
      **After:** 1.1. **Check:** `git diff docs/DECISIONS.md` is additive only.

## 2. `src/inference_x/schemas/chat.py`

- [x] 2.1 Add `ResponseWarning` with `type: Literal["substituted", "degraded"]`,
      `code: str`, `message: str`, `field: str | None = None`.
      **Why:** `strict` keys on the same condition that emits the warning (DEC-052 §2);
      a string cannot be that predicate's output. **After:** 1.1.
      **Check:** the `type` literal has exactly two members.
- [x] 2.2 Add `ResolvedRequest` mirroring `ChatCompletionRequest` minus `messages`,
      `stream`, `stream_options`, `strict`.
      **Why:** the derivability rule makes the block's contents derivable rather than
      negotiable. **After:** 2.1. **Check:** fields are exactly `model`, `temperature`,
      `max_tokens`, `top_p`, `max_context_tokens`, `max_output_tokens`, `priority`,
      `seed`.
- [x] 2.3 Append `strict: bool = False` to `ChatCompletionRequest`, **last**.
      **Why:** OS-2 R4 pins field ownership and append order across OS-2/OS-3/OS-4.
      **After:** 2.2. **Check:** `strict` is the final field; `seed` is unmoved.
- [x] 2.4 Add `resolved: ResolvedRequest | None = None` and
      `warnings: list[ResponseWarning] = Field(default_factory=list)` to
      `ChatCompletionResponse`.
      **Why:** plan §9 C.6 — the client determines substitutions from the response
      alone. Defaults keep direct construction in tests and engines working.
      **After:** 2.2. **Check:** existing `ChatCompletionResponse(...)` call sites
      compile unchanged.
- [x] 2.5 Do **not** touch `ChatStreamChunk`.
      **Why:** C15 — the Engine Boundary's streaming element type is not widened by
      this change. **After:** 2.4. **Check:** `ChatStreamChunk` still has exactly
      `content`, `finish_reason`, `usage`.

## 3. `src/inference_x/engines/base.py`

- [x] 3.1 Declare `count_prompt_tokens(self, request) -> int | None`, **non-abstract**,
      body `return None`, docstring stating that `None` means unavailable and callers
      must fail open.
      **Why:** DEC-047 §3 names the capability durable; non-abstract keeps
      "unavailable" representable, which §9 A.3 and DEC-047 §4 require. **After:** 2.1.
      **Check:** every existing `BaseEngine` subclass still instantiates.
- [x] 3.2 Do **not** declare `kv_capacity_tokens`.
      **Why:** DEC-047 §3 marks it provisional and forbids freezing it; Phase B4 may
      delete it. **After:** 3.1. **Check:** C9 — `BaseEngine`'s declared members are
      exactly `generate`, `generate_stream`, `is_healthy`, `count_prompt_tokens`.
      (Grepping the file for the name is too crude: the docstring names it in order
      to say it is *not* declared.)
- [x] 3.3 Update the class docstring's contract list to name the new declared method.
      **Why:** the docstring is the boundary's own description of what it guarantees.
      **After:** 3.1. **Check:** the DEC-047/DEC-049 paragraphs are unchanged.

## 4. `src/inference_x/routing/admission.py`

- [x] 4.1 Add `StrictModeViolationError(ValueError)` with a docstring stating it maps
      to 400 through the existing sanitized handler.
      **Why:** reuses `api/errors.value_error_handler` with no new registration, the
      same route `ContextTooLongError` takes. **After:** 2.3.
      **Check:** `api/errors.py` is unmodified.
- [x] 4.2 Widen `AdmissionResult` with `warnings: tuple[ResponseWarning, ...] = ()`.
      **Why:** admission decides the substitutions, so admission produces the warnings;
      the dataclass is frozen, hence a tuple. **After:** 4.1.
      **Check:** the dataclass is still `frozen=True`.
- [x] 4.3 Replace `_estimate_prompt_tokens`'s `getattr` probe with a direct
      `engine.count_prompt_tokens(request)` call; on `None`, use the chars/4 heuristic
      and emit `degraded` / `prompt_tokens_estimated` (field `messages`).
      **Why:** DEC-047 §3 hygiene, and §9 A.3 authorizes the heuristic to remain as a
      gate input while C.8 makes its use observable. **After:** 3.1, 4.2.
      **Check:** `_CHARS_PER_TOKEN_FALLBACK` is still used; the count is never a
      reported figure.
- [x] 4.4 Narrow `admit()`'s `engine` parameter from `Any` to `BaseEngine`.
      **Why:** calling a declared method through an `Any` parameter is a probe wearing
      a method's clothes. **After:** 4.3. **Check:** `mypy src/` passes and the
      DEC-048 baseline has not grown.
- [x] 4.5 Emit `degraded` / `sequence_gate_skipped` when `_effective_max_num_seqs`
      returns `None`, and `degraded` / `kv_gate_skipped` when
      `getattr(engine, "kv_capacity_tokens", None)` is `None`.
      **Why:** §9 C.8 — a gate that silently did not run is indistinguishable from one
      that ran and passed. **After:** 4.2. **Check:** both gates still fail open; the
      `getattr` at line 212 is byte-identical.
- [x] 4.6 Emit `substituted` / `max_tokens_clamped_to_context` and `substituted` /
      `max_tokens_clamped_to_kv_budget` (field `max_tokens`) at the two existing clamp
      sites, and raise `StrictModeViolationError` **inside the same branch** when
      `request.strict`.
      **Why:** DEC-052 §2 — one predicate, two outcomes. A second `if` would drift.
      **After:** 4.1, 4.5. **Check:** no `strict` check exists outside a `substituted`
      branch; `grep -c 'request.strict' src/inference_x/routing/admission.py` equals
      the number of `substituted` codes.
- [x] 4.7 Write a structured log record for every warning emitted, at the site that
      emits it.
      **Why:** plan §3.4(c) — one mechanism, two sinks. Two independent mechanisms
      guarantee the second rewrites the first. **After:** 4.6.
      **Check:** every `ResponseWarning` construction is adjacent to a `logger` call.
- [x] 4.8 Update the module docstring: gates 1–3 no longer discover
      `count_prompt_tokens` by `getattr`, and degradation is now typed.
      **Why:** lines 23 and 85–87 become false statements otherwise. **After:** 4.3.
      **Check:** the fail-open paragraph is preserved verbatim in meaning.
- [x] 4.9 Do **not** change any admission *decision*.
      **Why:** DEC-047 §4 forbids tightening to fail-closed in this milestone.
      **After:** 4.6. **Check:** C7 — `effective_max_tokens` and `reserved_tokens` are
      unchanged for every pre-existing test case.

## 5. `src/inference_x/services/chat_service.py`

- [x] 5.1 In `complete()`, build `ResolvedRequest` from `effective_request` and attach
      it plus `admitted.warnings` to the engine's response via `model_copy(update=...)`.
      **Why:** C16 — no engine constructs `resolved`; the Engine Boundary does not
      learn about admission. **After:** 2.4, 4.2. **Check:** `engines/vllm_engine.py`
      contains no reference to `resolved` or `warnings`.
- [x] 5.2 In `stream_response()`, emit exactly one pre-generation event via the
      existing `_event()` helper, after `admit()` and **before** the engine generator
      is consumed, carrying `choices: []`, `resolved` and `warnings`.
      **Why:** DEC-053 — the facts are determined at the anchor, and emitting them as a
      trailer loses them on the timeout path. **After:** 5.1.
      **Check:** the event is emitted before `gen = engine.generate_stream(...)` is
      iterated, and unconditionally, not gated on any request flag.
- [x] 5.3 Build `ResolvedRequest` from `effective_request` in both paths, never from
      `request`.
      **Why:** building from the original reports what the client asked for, inverting
      the point of the block. **After:** 5.2. **Check:** a clamped request's
      `resolved.max_tokens` equals the clamped value in both paths.
- [x] 5.4 Emit nothing after the usage event, on any path.
      **Why:** C3 and the post-generation requirement. OpenAI documents the usage
      chunk as the one streamed before `[DONE]`, and clients use it as an end sentinel.
      **After:** 5.2. **Check:** the ordered-sequence test in 8.1.
- [x] 5.5 Update `stream_response()`'s docstring event list to include the
      pre-generation event as item 0.
      **Why:** the docstring currently states the OS-2 order as normative. **After:**
      5.2. **Check:** it cites DEC-053 alongside DEC-049.

## 6. `src/inference_x/observability/middleware.py`

- [x] 6.1 Anchor TTFT to the first event carrying non-empty `delta.content` instead of
      the first raw chunk (`middleware.py:264`).
      **Why:** left alone, the pre-generation event makes every `/v1/metrics` TTFT
      include admission latency and become incomparable with figures recorded before
      this change — a DEC-050-class discontinuity introduced by accident. **After:**
      5.2. **Check:** C11 — TTFT for an identical generation is unchanged.
- [x] 6.2 Do **not** add any timing field, metric, or exporter.
      **Why:** Phase B2/B3. This task preserves an existing measurement's meaning; it
      does not create one. **After:** 6.1. **Check:** `schemas/metrics.py` is
      unmodified.
- [x] 6.3 Leave `_extract_sse_usage` unchanged.
      **Why:** it already returns `None` for a payload without a dict `usage`, so the
      pre-generation event contributes no token figure (C4). **After:** 6.1.
      **Check:** `git diff` shows no change to that function.

## 7. `src/inference_x/engines/vllm_engine.py`

- [x] 7.1 Update `count_prompt_tokens`'s docstring (lines 469–477): it is now a
      declared `BaseEngine` method, not a `getattr`-discovered optional attribute.
      **Why:** the sentence "BaseEngine doesn't declare this" becomes false.
      **After:** 3.1. **Check:** the diff for this file is docstring-only.
- [x] 7.2 Do **not** change the method's internal chars/4 fallback, its signature, or
      `kv_capacity_tokens`.
      **Why:** the engine's own fallback contract is outside this change's scope; the
      residual gap is recorded in `proposal.md` under Known limitation. **After:** 7.1.
      **Check:** `git diff --stat` shows no logic change in this file.

## 8. Tests

- [x] 8.1 `tests/unit/test_chat_service.py` — extend the OS-2 ordering tests to assert
      the **complete ordered event sequence** including the pre-generation event, for
      four cases: default, `include_usage: true`, timeout, and warnings present.
      **Why:** this test is the protocol in executable form; it must assert order and
      exact count, not substring presence. **After:** 5.4.
      **Check:** the three existing OS-2 tests still exist and still pass.
- [x] 8.2 `tests/unit/test_chat_service.py` — assert the pre-generation event precedes
      every content event, and is still emitted on the timeout path.
      **Why:** DEC-053's determinacy claim is only real if the emission point is tested.
      **After:** 8.1. **Check:** the timeout test asserts pre-generation, error,
      `[DONE]` and nothing else.
- [x] 8.3 `tests/unit/test_chat_service.py` — DEC-052 §3: for a request accepted under
      both `strict` values with an identical seed, assert byte-identical content and
      identical `resolved`.
      **Why:** this is the CI-checkable form of "strict never changes an accepted
      execution". **After:** 8.1.
      **Check:** the assertion compares content bytes, not lengths.
- [x] 8.4 `tests/unit/test_admission.py` — make `_FakeEngine` subclass `BaseEngine`.
      **Why:** 4.4 narrows the parameter type; a plain class would make the annotation
      dishonest. **After:** 4.4. **Check:** the existing admission tests pass unchanged
      otherwise.
- [x] 8.5 `tests/unit/test_admission.py` — DEC-052 §2: assert the set of conditions
      raising `StrictModeViolationError` **equals** the set emitting a `substituted`
      warning, by enumeration.
      **Why:** sampling would pass while the two sets drift. **After:** 8.4.
      **Check:** the test iterates codes rather than asserting two hand-picked cases.
- [x] 8.6 `tests/unit/test_admission.py` — for each `degraded` condition, assert the
      request is admitted under **both** `strict` values with the pre-change
      `effective_max_tokens`.
      **Why:** C8 and DEC-047 §4 — degradation must never become rejection. **After:**
      8.4. **Check:** all three degraded codes are covered.
- [x] 8.7 `tests/unit/test_admission.py` — assert an unclamped, capability-complete
      request yields `warnings == ()`.
      **Why:** truthfulness contracts fail silently toward never firing; only the empty
      case catches a warning path that quietly stopped being populated. **After:** 8.4.
      **Check:** asserts empty, not absent and not falsy.
- [x] 8.8 `tests/unit/test_schemas.py` — derivability test: `ResolvedRequest`'s field
      set equals `ChatCompletionRequest`'s minus `{messages, stream, stream_options,
      strict}`.
      **Why:** enforces the rule in CI rather than in review, so a future field forces
      an explicit decision. **After:** 2.4. **Check:** the test fails if either schema
      gains a field.
- [x] 8.9 `tests/unit/test_schemas.py` — assert `strict` defaults to `false`, `resolved`
      defaults to `None`, `warnings` defaults to an empty list, and a non-boolean
      `strict` is a validation error.
      **Why:** C1 and C2 — additive with defaults, no behaviour change for existing
      clients. **After:** 2.4. **Check:** covers all three defaults.
- [x] 8.10 `tests/unit/test_routes.py` — assert `strict: true` on a clamping request
      returns 400 through the existing handler, and that the default returns 200 with a
      `substituted` warning.
      **Why:** §9 C.7 requires both paths covered. **After:** 4.6.
      **Check:** the 400 body is the sanitized shape `value_error_handler` produces.
- [x] 8.11 `tests/unit/test_observability.py` — assert TTFT is measured from the first
      content event and that the pre-generation event does not enter it.
      **Why:** C11 — the regression this task exists to prevent is silent. **After:**
      6.1. **Check:** the test feeds a stream whose pre-generation event is separated in
      time from the first content event.
- [x] 8.12 `tests/unit/test_observability.py` — assert `_extract_sse_usage` returns
      `None` for the pre-generation event.
      **Why:** C4 — no token figure may originate from it. **After:** 6.3.
- [x] 8.13 Consumer compatibility — assert `benchmarks/runner.py` and
      `playground/streaming.py` parse a stream containing the pre-generation event
      **without modification** to either file.
      **Why:** C5, C6 and C10. The tolerance was verified by reading `runner.py:75` and
      `streaming.py:63`, and a test is what keeps it true. **After:** 5.2.
      **Check:** `git diff` touches neither file.

## 9. Explicit do-not-touch (verify at self-review)

- [x] 9.1 **No Phase B work.** `engines/driver.py`, `observability/exporters.py`,
      `utils/vllm_pool_config.py` and `api/deps.py` are unmodified.
- [x] 9.2 **No Phase C work.** No replay handle, no trace id, no batch-invariance
      claim, no determinism claim anywhere in code or docs.
- [x] 9.3 **No AsyncLLM migration.** `EngineDriver` keeps its locking, dead-flag
      semantics (DEC-043) and restart policy.
- [x] 9.4 **No timings.** No timing field on any schema, no new metric, no exporter.
      Task 6.1 preserves an existing measurement and adds nothing.
- [x] 9.5 **No `inference_x/execution/`**, no backend-neutral DTO package, no second
      type system.
- [x] 9.6 **No second backend.** `ModelEntry.engine` remains `Literal["vllm"]`;
      `vllm` remains a required dependency, not an optional extra.
- [x] 9.7 **No `kv_capacity_tokens` declaration.** Both `getattr` sites
      (`routing/admission.py:212`, `api/routes/metrics.py:43`) are byte-identical.
- [x] 9.8 **No registry factory.** `engines/registry.py` stays at 0 bytes; the
      ARCHITECTURE.md gap remains a tracked open item and the doc is not weakened to
      match the code.
- [x] 9.9 **No scheduler or batching change.** No change to `max_num_seqs`,
      `max_num_batched_tokens`, block size, prefix caching, or any dispatch ordering.
- [x] 9.10 **No `playground/` change.** It is a client. Its two `len(content.split())`
      displays remain a recorded follow-up, not OS-4 work.
- [x] 9.11 **No `ChatStreamChunk` widening** and no change to
      `BaseEngine.generate_stream`.

## 10. Validation and self-review

- [x] 10.1 Every Compatibility Invariant C1–C16 holds.
- [x] 10.2 Only Intentional Behavioral Changes B1–B7 landed.
- [x] 10.3 The DEC-048 mypy baseline has not grown (C14).
- [x] 10.4 `uv run ruff check .` and `uv run mypy src/` pass.
- [x] 10.5 Full unit suite passes; the xfail count is unchanged from 0.3.
- [x] 10.6 `git diff --stat` touches no file outside `proposal.md`'s Repository impact.
- [x] 10.7 `openspec validate --strict` passes for this change.
- [x] 10.8 Do not mark complete if any acceptance criterion in `proposal.md` is unmet.
