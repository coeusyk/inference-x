## 1. Fix (BaseEngine-exclusive)

- [x] 1.1 In `src/inference_x/engines/base.py`, remove `async` from `generate_stream`'s
      `@abstractmethod` declaration. The parameter list, return annotation
      (`AsyncGenerator[ChatStreamChunk, None]`), and docstring do not change.
- [x] 1.2 Do not add a `yield` or a `raise NotImplementedError` to the abstract body —
      the declaration form alone is the fix; the body stays docstring-only, consistent
      with every other method on this ABC.
- [x] 1.3 Do not touch `vllm_engine.py`, `driver.py`, `chat_service.py`, or any concrete
      implementation or test stub — they already match the corrected declaration exactly
      (sandbox-verified: every implementation is already a true async generator
      function).

## 2. Baseline

- [x] 2.1 In `pyproject.toml`, remove `"inference_x.services.chat_service"` (and its
      trailing comment) from the `[[tool.mypy.overrides]]` module list that sets
      `ignore_errors = true`.
- [x] 2.2 Leave the `vllm_engine`, `driver`, and `deps` baseline entries untouched — none
      of their remaining errors involve `generate_stream`.

## 3. Validation

- [x] 3.1 Reproduce (pre-fix): with the current declaration, `mypy` run unsuppressed
      against `services/chat_service.py` in isolation reports exactly 3 errors; against
      `engines/vllm_engine.py` in isolation reports exactly 9 errors, one of which
      (`:556`) is the `generate_stream` invalid-override error.
- [x] 3.2 Prove: removing `async` alone eliminates the override mismatch — with no other
      signature or body edits, the `:556` invalid-override error is gone.
- [x] 3.3 Prove: no implementation edits are required — `vllm_engine.py` and every test
      stub remain unmodified and satisfy the corrected declaration.
- [x] 3.4 Prove: no caller edits are required — ChatService, EngineDriver, middleware,
      routing, benchmarks, and tests remain valid without adaptation.
- [x] 3.5 Prove: the removed mypy-baseline entry is exactly
      `inference_x.services.chat_service` (the audit-identified entry); no other baseline
      entry is removed or grown.
- [x] 3.6 Confirm (post-fix): unsuppressed mypy reports 0 errors for `chat_service.py`
      and 8 errors for `vllm_engine.py` — override gone, unrelated 8 unchanged.
- [x] 3.7 `mypy src/` clean project-wide with the `chat_service` baseline entry removed.
- [x] 3.8 Regression guard: future `BaseEngine.generate_stream` implementations MUST
      type-check without requiring suppression for this contract (documented in design /
      spec; no suppression path is introduced by this change).
- [x] 3.9 `ruff check .` clean.
- [x] 3.10 `pytest tests/unit/` — all existing tests pass unchanged; no new test is
      required since no executable behavior changes.
- [x] 3.11 `openspec validate close-generate-stream-contract-gap --strict` passes.

## Out of scope checklist (must remain undone)

- [ ] Confirm no runtime behavior changed — no concrete method's bytecode differs.
- [ ] Confirm no `ChatStreamChunk` change.
- [ ] Confirm no `EngineDriver` change.
- [ ] Confirm no Engine Boundary widening (no `kv_capacity_tokens` promotion, no
      `create_engine`/factory work).
- [ ] Confirm no `inference_x/execution/` package was introduced.
- [ ] Confirm no streaming protocol or `observability/middleware.py` change.
- [ ] Confirm no AsyncLLM or other Phase B implementation work.
- [ ] Confirm `docs/DECISIONS.md` was not edited by this change.
- [ ] Confirm the `vllm_engine`, `driver`, and `deps` baseline entries were left
      untouched.
- [ ] Confirm Compatibility Invariants hold: same callable type; no caller changes.
