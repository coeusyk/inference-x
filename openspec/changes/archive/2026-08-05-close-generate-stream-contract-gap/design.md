## Context

Python distinguishes an *async generator function* from a plain *coroutine function* by
whether the function body contains `yield`, not by anything in its signature or
annotation. `engines/base.py:39-47` is:

```python
@abstractmethod
async def generate_stream(
    self, request: ChatCompletionRequest
) -> AsyncGenerator[ChatStreamChunk, None]:
    """Run inference and yield chunks as they are generated (DEC-049)....
    """
```

No `yield` appears in the body (it is `@abstractmethod`, a docstring only), so this is a
coroutine function whose *declared* return type happens to be `AsyncGenerator[...]`.
Calling it returns a coroutine that, if awaited, would resolve to an async generator —
mypy reports this precisely as `Coroutine[Any, Any, AsyncGenerator[ChatStreamChunk,
None]]`. Every concrete override (`VLLMEngine.generate_stream`, every test stub in
`tests/unit/`) has `yield` in its body and so *is* an async generator function — calling
it returns the async generator immediately, no `await` involved. Every caller
(`ChatService.stream_response:149`, `AdmissionController`) calls and iterates
immediately. The ABC's declared type describes neither side of its own contract.

## Compatibility Invariants

1. **Same callable type.** The abstract declaration and every implementation MUST
   denote the same callable type. This is stronger than "no invalid override": it is a
   durable type-identity requirement that survives future engines.
2. **No caller changes are permitted.** ChatService, EngineDriver, middleware, routing,
   benchmarks, and tests must remain valid without adaptation.

## Goals / Non-Goals

**Goals:**
- Correct the abstract contract: `BaseEngine.generate_stream`'s declared type matches
  what every implementation returns and what every caller expects (Compatibility
  Invariant 1).
- Remove the `services.chat_service` mypy baseline entry once it type-checks clean —
  exactly the entry identified by the audit; no other baseline entry is removed or grown.
- Leave every caller valid without adaptation (Compatibility Invariant 2).
- No runtime behavior change of any kind.
- No streaming behavior change.
- No Engine Boundary ownership change.
- No `ChatStreamChunk` change.
- No `EngineDriver` change.
- No admission change.

**Non-Goals:**
- No AsyncLLM work of any kind.
- No `engines/driver.py` deletion.
- No `create_engine`/factory work (DEC-047 §5 already defers this past roadmap B1).
- No `inference_x/execution/` package — forbidden outright by the existing "Boundary is
  not widened further" requirement in `openspec/specs/platform/spec.md`.
- No streaming protocol changes — DEC-049/DEC-053 event order, content, and count are
  untouched.
- No `observability/middleware.py` changes.
- No Phase B implementation of any kind.
- No typing cleanup outside this one contract — the `vllm_engine`, `driver`, and `deps`
  baseline entries stay exactly as they are; their remaining errors are pre-existing and
  unrelated (vLLM's untyped `LLM`, `Optional` narrowing pending the Phase B1 driver
  deletion, composition-root argument types).

## Decisions

### Drop `async`; do not add a `yield` to the abstract body

The fix is to declare `generate_stream` as a plain (non-`async`) function whose return
annotation is `AsyncGenerator[ChatStreamChunk, None]`. This is the standard way to type an
abstract method whose concrete implementations are async generator functions: calling an
async generator function returns the `AsyncGenerator` object synchronously — the
annotation is describing what calling the method *returns*, and a plain `def` correctly
models "calling this returns an async generator, not something you await first." mypy
then treats a concrete `async def ...: yield ...` override as satisfying that same
`AsyncGenerator[...]` return type — no invalid-override relationship, because both sides
now describe the same thing.

Alternative considered: keep `async def` and add a `yield` somewhere unreachable in the
abstract body (e.g., after a `raise NotImplementedError`) so Python's own runtime also
classifies it as an async generator function. Rejected — this requires dead code in the
abstract body, is inconsistent with how every other method on this ABC is declared
(`count_prompt_tokens`, `is_healthy` — docstring-only bodies, no sentinel `raise`), and
buys nothing the plain-`def` form doesn't already provide. The plain-`def` form is also
the one verified in the sandbox to clear exactly the errors in scope and introduce none.

### Only the `chat_service` baseline entry is removed

`vllm_engine`'s baseline entry keeps 8 unrelated errors (all pre-existing, all about
vLLM's untyped `LLM` class) after this fix — it stays in the baseline, just with a lower
true error count than before. Removing an entry from the DEC-048 baseline list is only
correct when a module has zero unsuppressed errors; `chat_service` reaches zero,
`vllm_engine` does not. `driver` and `deps` are untouched entirely — neither module's
error count is affected by this declaration change.

## Risks / Trade-offs

None functional — sandbox-verified as a mypy-only change (see proposal.md's verification
section for exact before/after error counts). The only risk is process: if a future
change reintroduces a `chat_service` mypy error, there is no longer a baseline entry to
mask it, so `mypy src/` will fail loudly instead of silently passing — this is the
intended effect of DEC-048 §4 ("over-suppression is a defect, not a safe default"), not a
regression.

## Migration Plan

1. In `engines/base.py`, remove `async` from `generate_stream`'s declaration. No other
   character of the signature, return annotation, or docstring changes. No
   implementation file is edited. No caller file is edited.
2. In `pyproject.toml`, remove the `"inference_x.services.chat_service"` line (and its
   trailing comment) from the `[[tool.mypy.overrides]]` block that lists baselined
   modules — exactly the entry identified by the audit. Leave `vllm_engine`, `driver`,
   and `deps` entries exactly as they are.
3. Prove the validation properties below. Run `mypy src/` — expect clean, project-wide,
   with one fewer baseline entry than before.
4. Run `ruff check .` and `pytest tests/unit/` — expect both unchanged, since nothing
   executable changed.

## Validation properties (required)

Implementation validation MUST prove all of the following:

1. **Removing `async` alone eliminates the override mismatch.** With no other signature
   or body edits, the `generate_stream` invalid-override error disappears.
2. **No implementation edits are required.** Concrete engines and test stubs are
   unchanged and already satisfy the corrected declaration.
3. **No caller edits are required.** ChatService, EngineDriver, middleware, routing,
   benchmarks, and tests remain valid without adaptation.
4. **The removed mypy-baseline entry is exactly the one identified by the audit** —
   `inference_x.services.chat_service`. No other baseline entry is removed or grown.

## Phase B carry-forward / regression guard

Roadmap B1 (AsyncLLM) implements `BaseEngine` directly and will write its
`generate_stream` against whatever declaration exists at that time. After this change,
B1's implementation is checked against a sound contract from the start, instead of
inheriting an invalid-override relationship the moment it's written. This does not reduce
B1's scope or pre-decide anything about B1's design — it only ensures B1 starts from a
contract that type-checks.

**Regression guard:** future implementations of `BaseEngine.generate_stream` MUST
type-check without requiring suppression (no mypy baseline entry, no `# type: ignore`,
and no other suppression for this contract).
