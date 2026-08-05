# Tasks — OS-3 Deterministic generation

Implement in the order below. Re-read proposal Compatibility Invariants C1–C15
and Intentional Behavioral Change B1 before coding. Do not expand scope.

## 0. Preconditions

- [x] 0.1 Confirm OS-1 CI gate is green on the branch point (`pytest`, `ruff`,
      `mypy`) so later failures are attributable to this change.
- [x] 0.2 Re-read the Deterministic Generation Contract (G1–G5 / N1–N8) and
      ownership one-liner in `proposal.md`.
- [x] 0.3 Confirm OS-2 is archived and Engine Boundary will not be edited
      (`engines/base.py` stays untouched — C4).

## 1. Decision record

- [x] 1.1 Add **DEC-051** to `docs/DECISIONS.md` with status `accepted`:
      - Title: Seed support / Deterministic Generation Contract (OS-3)
      - Record G1–G5 and N1–N8 (or reference the OpenSpec Contract section)
      - Ownership one-liner: *The client owns requesting determinism; the
        backend owns honouring it; the runtime must not invent it.*
      - Forward-unchanged including `-1`; no runtime normalization
      - Echo deferred to OS-4; batch invariance deferred to Phase C3
      - Silent-drop of client `seed` is superseded as a failure mode
- [x] 1.2 Do not reopen DEC-047 or rewrite DEC-049.

## 2. `schemas/chat.py`

- [x] 2.1 Append `seed: Optional[int] = None` (Field with description) **after**
      `stream_options` on `ChatCompletionRequest`. No `ge=0` (R2).
- [x] 2.2 Description MUST state the value is forwarded unchanged to the live
      sampler when set; MUST NOT claim end-to-end determinism (G5).
- [x] 2.3 Do **not** modify `ChatCompletionResponse` (C2).
- [x] 2.4 Do **not** add `strict` — OS-4.

## 3. `engines/vllm_engine.py`

- [x] 3.1 In `_sampling_params` only: when `request.seed is not None`, set
      `kwargs["seed"] = request.seed` (exact int, including `-1`).
- [x] 3.2 When `request.seed is None`, omit the `seed` key entirely (R1, G3).
- [x] 3.3 Do **not** map `-1` → `None`, reject negatives, or emit warnings (R2, N8).
- [x] 3.4 Do **not** edit `generate` / `generate_stream` beyond what sharing
      `_sampling_params` already provides (G4).
- [x] 3.5 Do **not** change the degraded `_VLLM_AVAILABLE == False` path (C10).
- [x] 3.6 Do **not** touch `engines/base.py` or `engines/driver.py` (C4, C5).

## 4. Unit tests — forwarding and schema

- [x] 4.1 Extend `tests/unit/test_vllm_sampling.py`:
      - `seed=42` → kwargs `seed == 42`
      - `seed=-1` → kwargs `seed == -1` (unchanged forward)
      - omitted `seed` → `"seed" not in kwargs`
- [x] 4.2 Add schema (or route) coverage that an integer `seed` validates and a
      non-integer `seed` yields 422 / validation error.

## 5. Unit tests — identity and concurrency

- [x] 5.1 Identity test: two completions with **same prompt**, **same
      parameters** (including `temperature=0`), **same seed**, **sequential
      execution** (one fully completed before the next starts; single
      in-flight) → identical assistant content. Use existing GPU/live skip
      patterns if the environment requires them; do not weaken the sequential
      requirement.
- [x] 5.2 Concurrency test: same seed under multi-in-flight marked **`xfail`**,
      reason cites **Phase C3** (batch-composition nondeterminism).
- [x] 5.3 Place tests in `tests/unit/` (extend sampling module or add a focused
      module). Do not invent an oracle suite (Phase C4).

## 6. Documentation language

- [x] 6.1 Update `docs/UNDERSTANDING-INFERENCE-X.md` (and optionally `README.md`)
      so the gap statement no longer claims seed is never passed; say seed is
      **honoured** / reaches the sampler.
- [x] 6.2 Do **not** write that runs are reproducible or that the server is
      deterministic (G5, R4).

## 7. Explicit do-not-touch (verify at self-review)

- [x] 7.1 Confirm untouched: `engines/base.py`, `engines/driver.py`,
      `services/chat_service.py` SSE ordering, `routing/admission.py`,
      `benchmarks/runner.py`, `observability/middleware.py`, `api/deps.py`,
      `engines/registry`.
- [x] 7.2 Confirm no `warnings` / `resolved` / `strict` / `count_prompt_tokens`.
- [x] 7.3 Confirm default benchmark requests remain unseeded (C8).

## 8. Validation and self-review

- [x] 8.1 Every Compatibility Invariant C1–C15 holds.
- [x] 8.2 Only Intentional Behavioral Change B1 landed.
- [x] 8.3 DEC-048 mypy baseline has not grown (C12).
- [x] 8.4 `ruff check .` and `mypy src/` pass.
- [x] 8.5 Full unit suite passes.
- [x] 8.6 `openspec validate --strict` passes for this change.
- [x] 8.7 Do not mark complete if any acceptance criterion in `proposal.md` is
      unmet.
