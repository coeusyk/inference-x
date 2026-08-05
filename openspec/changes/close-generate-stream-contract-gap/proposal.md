## Why

`BaseEngine.generate_stream` (`engines/base.py:39-47`) is declared:

```python
async def generate_stream(
    self, request: ChatCompletionRequest
) -> AsyncGenerator[ChatStreamChunk, None]:
```

The `@abstractmethod` body carries only a docstring — no `yield` — so this is an
`async def` with no `yield` in its body: a coroutine function, not an async generator
function. mypy therefore reads its return type as
`Coroutine[Any, Any, AsyncGenerator[ChatStreamChunk, None]]` ("await me, then iterate")
rather than `AsyncGenerator[ChatStreamChunk, None]` directly. Every concrete
implementation (`VLLMEngine.generate_stream`, every test stub) is a true async generator
function (`async def` with `yield` in the body), and every caller
(`ChatService.stream_response`, `AdmissionController`) calls it and iterates immediately,
never `await`s it first. The declared type contradicts both sides of its own contract.

Verified fresh in a disposable sandbox this session (`src/` copied out, current
declaration and current suppressions left untouched in the real repo):

- `services/chat_service.py` in isolation, unsuppressed: **3 errors** — an
  "Incompatible types in assignment" at the `gen = engine.generate_stream(...)` call site
  plus two "`None` has no attribute `__anext__`" errors that follow from it (mypy narrows
  `gen`'s type to include `None` because it thinks the assignment can fail).
- `engines/vllm_engine.py` in isolation, unsuppressed: **9 errors** — 8 are pre-existing
  and unrelated (vLLM's `LLM` class is untyped; `dict-item`, `valid-type`, `attr-defined`,
  `arg-type` — none of these involve `generate_stream`). The 9th, at `:556`, is exactly
  this contract: *"Return type `AsyncGenerator[...]` of `generate_stream` incompatible
  with return type `Coroutine[..., AsyncGenerator[...]]` in supertype `BaseEngine`"* — an
  invalid-override error.
- `engines/base.py` in isolation: clean, both before and after.
- Applying the fix (dropping `async` from the declaration, nothing else) and re-running:
  `chat_service.py` → 0 errors. `vllm_engine.py` → 8 errors, the `:556` override error
  gone, the 8 unrelated errors unchanged. The fix removes exactly 4 errors and introduces
  none.

Both are currently invisible in `mypy src/`: `inference_x.services.chat_service` and
`inference_x.engines.vllm_engine` both sit in the DEC-048 mypy baseline
(`pyproject.toml`, `ignore_errors = true`). DEC-049's migration notes named clearing the
`chat_service` baseline entry as an exit condition once the streaming contract
type-checks; OS-2 widened the chunk *payload* (`str` → `ChatStreamChunk`) but never
corrected the *declaration*, so the entry survived under a different set of error
messages.

## What Changes

- `engines/base.py`: drop `async` from `generate_stream`'s `@abstractmethod` declaration.
  The signature, parameter, return annotation, and docstring are otherwise unchanged. No
  `yield` is added to the abstract body.
- `pyproject.toml`: remove `"inference_x.services.chat_service"` from the DEC-048 mypy
  `ignore_errors` module list. The other three entries (`vllm_engine`, `driver`, `deps`)
  are untouched — their remaining errors are unrelated to this contract.

## Capabilities

### Modified Capabilities
- `platform`: `BaseEngine.generate_stream`'s declared type now matches how every
  implementation and every caller already uses it. No behavior, payload, or event-order
  change.

### Compatibility Invariants
- The abstract declaration and every implementation MUST denote the same callable type.
- No caller changes are permitted. ChatService, EngineDriver, middleware, routing,
  benchmarks, and tests must remain valid without adaptation.
- Future implementations of `BaseEngine.generate_stream` MUST type-check without
  requiring suppression.

## Impact

- Code: `src/inference_x/engines/base.py` (declaration only) and `pyproject.toml`
  (baseline list only). No other file — no implementation edits, no caller edits.
- Runtime behavior: none. Sandbox-verified — the change is mypy-only; no concrete
  method's bytecode differs.
- Tests: no new runtime test needed, since nothing executable changes. Validation must
  prove: removing `async` alone eliminates the override mismatch; no implementation
  edits required; no caller edits required; the removed mypy-baseline entry is exactly
  `inference_x.services.chat_service` (the audit-identified entry). Plus the existing
  `ruff`/`pytest` gate confirming no regression.
- Docs: none edited by this change. A one-line DEC-049 status addendum was previously
  suggested (Pre-Phase-B execution plan §2) but is explicitly out of scope here per this
  change's own constraints — `docs/DECISIONS.md` is not touched.
