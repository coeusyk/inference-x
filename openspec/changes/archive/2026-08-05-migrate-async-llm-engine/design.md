# Design: migrate-async-llm-engine

## Context

`EngineDriver` (`engines/driver.py`) owns a synchronous `vllm.LLM.llm_engine`
exclusively and is the sole caller of `add_request`/`step()`, demultiplexing
`step()` output to the right request by id. This exists to fix a specific,
reproduced race (DEC-038: a one-shot `finished=True` handoff loses output when
a *different* thread's `step()` call surfaces your request's terminal output;
DEC-039: the driver-thread fix; DEC-043: the `_dead_lock` atomicity fix for a
submit-vs-broadcast race).

vLLM 0.22.1 ships `AsyncLLM` (`vllm/v1/engine/async_llm.py`), a v1-engine async
client that already solves per-request delivery, cancellation, and health
signaling without a driver thread. `PHASE-A-ARCHITECTURE.md` §10 names
replacing `LLM` + `EngineDriver` with `AsyncLLM` as B1, scoped strictly to the
`VLLMEngine` internals — `BaseEngine`'s contract, `ChatService`, and
`AdmissionController` are out of scope.

## Goals / Non-Goals

**Goals**
- Delete `EngineDriver` and its DEC-038/039/043 race-class history.
- Preserve every existing `BaseEngine` caller contract with zero caller edits.
- Preserve the DEC-050 single-source guarantee for terminal usage metadata.
- Make cancellation and health real signals sourced from `AsyncLLM`, not
  reimplemented.

**Non-Goals**
- Splitting multi-model serving into separate processes (B6).
- Rescoping admission control or the KV gate (B4).
- Redesigning `pool_size` semantics for `pool_size > 1` — see Decision 3.
- Any change to `BaseEngine`, `ChatService`, `AdmissionController`, or route
  handlers.
- Prometheus / native stat-logger passthrough (B2).

## Decisions

### Decision 1 — `_POOL_STEP_LOCK` is deleted, not relocated

`_POOL_STEP_LOCK` (`vllm_engine.py:35`) serializes concurrent `EngineDriver`
threads calling `step()` on separate sync engines sharing one process ("vLLM
V1 forward context", DEC-039). `AsyncLLM.__init__` constructs its own
`EngineCoreClient.make_async_mp_client(...)` (`async_llm.py:146`) — each
`AsyncLLM` instance owns an independent background engine-core **process**,
not a shared in-process step loop. There is no shared resource left for
`_POOL_STEP_LOCK` to guard. It is deleted, not carried forward under a new
name.

This does **not** mean multiple co-located `AsyncLLM` instances are proven
safe — only that the specific mechanism the old lock guarded against has no
equivalent here. See Decision 3.

### Decision 2 — `generate()` is derived from `generate_stream()`

`VLLMEngine.generate()` (non-streaming) is implemented in terms of
`generate_stream()` (draining it and discarding intermediate chunks, or
delegating to a shared internal primitive that produces the same terminal
metadata) rather than making a second, independent call into
`AsyncLLM.generate()`.

Rejected alternative: an independent `AsyncLLM.generate()` call per path.
Rejected because it would require re-deriving terminal metadata a second time
outside `derive_terminal_metadata`, directly reopening the DEC-050 drift the
helper exists to prevent, and would double the surface that must implement
`AsyncLLM`'s cancellation/timeout handling correctly (Decision 4).

The literal mechanism (iterate-and-discard vs. a shared internal primitive) is
an implementation detail. What must not change across future refactors is the
property: **exactly one code path calls into `AsyncLLM.generate()`**, and both
`VLLMEngine.generate()` and `VLLMEngine.generate_stream()` are expressed in
terms of it or a shared primitive derived from it. This is recorded as a
compatibility invariant below so a later "simplification" cannot silently
reintroduce a second call site.

### Decision 3 — `pool_size > 1`: not guaranteed, not forbidden

The prior review round concluded `_POOL_STEP_LOCK` has no equivalent under
AsyncLLM (Decision 1) and initially over-claimed this as "pool_size > 1 must
fail loudly." That claim was checked against the evidence and does not hold:
nothing inspected — not `PHASE-A-ARCHITECTURE.md`, not DEC-047, not
`async_llm.py` itself — demonstrates that `AsyncLLM` cannot safely support
multiple co-located instances. `async_llm.py` contains no
singleton/single-instance guard of any kind (verified: zero matches for
`singleton|only one|single instance|global|process-wide|not thread.safe|one
engine` in the file), and each instance owning its own subprocess (Decision 1)
is architecturally independent of any other instance.

**Frozen wording:**

> B1 preserves the existing public configuration surface (`pool_size`) but
> makes no architectural guarantees about values >1 until AsyncLLM
> multi-instance support is explicitly verified. B6 remains the only phase
> that may redesign multi-engine serving.

"Not guaranteed" is deliberately weaker than "forbidden": B1 must not add a
construction-time failure for `pool_size > 1` that the evidence doesn't
support, and must not silently assume it works either. Concretely: `VLLMEngine`
construction accepts `pool_size` unchanged; no new pool-size validation is
added by this change in either direction. If multi-instance behavior needs a
real answer (GPU memory partitioning, executor/device placement across
co-located `AsyncLLM` instances), that answer is out of scope for B1 and
belongs to whichever phase first needs it verified — most likely B6.

### Decision 4 — Cancellation is a compatibility invariant, not an implementation note

Verified directly against `async_llm.py`: `generate()`'s
`except (asyncio.CancelledError, GeneratorExit)` handler calls
`await self.abort(q.request_id, internal=True)`, and `abort()`
(`async_llm.py:709-721`) calls `await self.engine_core.abort_requests_async(...)`
— a real call into the engine-core process, not local bookkeeping.

This is a genuine behavioral change from `EngineDriver`, where an abandoned
consumer only stops *reading* — `_pending[request_id]` is removed solely on
`output.finished` (`driver.py:229-237`), so the underlying vLLM computation for
a disconnected client kept running today. Because this is a real improvement a
later "simplification" could silently regress (e.g. swallowing
`GeneratorExit` without propagating it to `AsyncLLM.generate()`, which would
turn real cancellation back into consumer-side abandonment), it is recorded as
a **compatibility invariant**, not left as an implementation note:

> Cancellation must remain observable as engine-side request abortion, not
> merely consumer-side abandonment.

### Decision 5 — Health: `errored` / `dead_error` replace `is_dead` / `EngineDriverDeadError`

Verified: `AsyncLLM.errored` (`async_llm.py:1045-1058`) is
`self.engine_core.resources.engine_dead or not self.is_running`;
`dead_error` returns `EngineDeadError()`; `check_health()`
(`async_llm.py:900-903`) raises `self.dead_error` when `errored`.
`VLLMEngine.is_healthy()` becomes `not self._llm.errored`.
`EngineDriverDeadError` (repo-local) is deleted along with `driver.py`;
vLLM's own `EngineDeadError` is used directly — no repo-local wrapper.

### Decision 6 — KV cache stats: same attribute chain, different root

Verified: `cache_config.num_gpu_blocks` is populated on the shared
`vllm_config` object at startup (`core_client.py:712-714`), and
`AsyncLLM.vllm_config` is a live instance attribute
(confirmed via `async_llm.py:306`'s use of
`self.vllm_config.cache_config.kv_sharing_fast_prefill`). `_log_kv_cache_stats()`
changes only its root object — `self._llm.vllm_config.cache_config` instead of
`self._llm.llm_engine.vllm_config.cache_config` — the `.num_gpu_blocks` /
`.block_size` attribute chain is unchanged.

### Decision 7 — Timeout: no native per-request timeout; wrapper becomes effective, not cosmetic

Verified: `generate()` takes no `timeout` parameter; no per-request timeout
construct exists in `async_llm.py` (`shutdown(timeout=...)` and
`wait_for_requests_to_drain(drain_timeout=...)` are unrelated — engine
shutdown and elastic-scaling drain, not per-request). `VLLMEngine` must still
supply its own timeout wrapper (`_COMPLETION_TIMEOUT_S`-equivalent) around
consumption of `generate_stream()`.

This is an **intentional behavioral change**, not a preserved one: today,
`future.result(timeout=...)` firing on `EngineDriver` just stops *waiting* —
the underlying vLLM computation for that request keeps running untouched until
natural completion. Under AsyncLLM, wrapping consumption in
`asyncio.wait_for(..., timeout=...)` delivers a `CancelledError` into the
awaiting `generate()`, which (Decision 4) now triggers a real `abort()`. A
timeout will, for the first time, actually stop the engine-side computation.
This must be called out in the platform-facing changelog as a behavioral
improvement, not silently absorbed as "same as before."

### Decision 8 — `derive_terminal_metadata`: relocated, not reimplemented (OWN-B5)

Per DEC-050 §3 and the Phase A audit's OWN-B5 finding, the function is moved
from `driver.py` into `vllm_engine.py` verbatim (signature and body
unchanged: it maps a finished vLLM `RequestOutput` to `(finish_reason, usage)`
by reading `output.outputs[0]`, `output.prompt_token_ids`,
`completion.token_ids` — nothing here is `EngineDriver`-specific). Both
`generate()` and `generate_stream()` continue to call this one function
(Decision 2 makes this structural, not conventional: there is only one call
site by construction once `generate()` is derived from `generate_stream()`).

## Compatibility Invariants

1. `BaseEngine`'s declared contract (`generate`, `generate_stream`,
   `count_prompt_tokens`, `kv_capacity_tokens`) is unchanged.
2. `ChatService`, `AdmissionController`, and every route handler require zero
   code changes — a required edit to any of them is a scope break.
3. Streamed and non-streamed usage continue to derive from exactly one
   function (`derive_terminal_metadata`, relocated per Decision 8).
4. Exactly one code path calls into `AsyncLLM.generate()` (Decision 2);
   `VLLMEngine.generate()` and `VLLMEngine.generate_stream()` are both
   expressed in terms of it.
5. Cancellation must remain observable as engine-side request abortion, not
   merely consumer-side abandonment (Decision 4).
6. `pool_size > 1` is not guaranteed and not forbidden by this change
   (Decision 3) — no new construction-time validation is added for it in
   either direction.
7. No new package (`inference_x/execution/` or similar) or dual type system is
   introduced — the existing "Boundary is not widened further" requirement
   (platform spec) continues to hold.

## Task 1.4 — verified conclusion

Whether `AsyncLLM` rejects a submission immediately and synchronously once the
engine is already dead was traced end to end and **confirmed** via two
independent, redundant synchronous guards — not one:

- **Guard A — `AsyncLLM.add_request()` itself** (`async_llm.py:300-301`), the
  first check in the method body, before any prompt processing or contact
  with `EngineCoreClient`:
  ```python
  if self.errored:
      raise EngineDeadError()
  ```
- **Guard B — the transport client, defense in depth.** `AsyncLLM.__init__`
  constructs an `AsyncMPClient` (or a `DPAsyncMPClient`/`DPLBAsyncMPClient`
  subclass) via `EngineCoreClient.make_async_mp_client(...)`
  (`core_client.py:108-128`, `async_llm.py:146`). Its
  `add_request_async` (`core_client.py:1090`) calls
  `await self._send_input(EngineCoreRequestType.ADD, request)` →
  `_send_input`/`_send_input_message` (`core_client.py:1032-1052`), whose
  first line is `self.ensure_alive()` (`core_client.py:1052`);
  `ensure_alive()` (`core_client.py:658-660`) raises `EngineDeadError()` if
  `self.resources.engine_dead`.

Propagation is clean, not orphaned: `generate()` has a dedicated
`except EngineDeadError:` branch, distinct from its
`except (asyncio.CancelledError, GeneratorExit):` branch, that logs and
re-raises immediately without calling `abort()` (nothing to abort — the
comment reads *"Engine is dead. Do not abort since we shut down."*).

**Conclusion: rejected immediately.** Not accepted-then-failed-later, not
orphaned — both guards fire synchronously, before any request state is
created on the engine-core side.

The spec delta's implementation-neutral MODIFIED requirement ("Engine driver
rejects requests immediately after failure") is satisfied by this backend —
confirmed by evidence, not assumed. The requirement's own wording stays
implementation-neutral (it does not name `AsyncLLM`, `errored`, or
`ensure_alive()`); this section is where the AsyncLLM-specific proof lives.

## Risks / Trade-offs

- **Risk**: `AsyncLLM`'s engine-core subprocess model means engine startup/
  shutdown timing changes (a real subprocess vs. an in-process object).
  `deps.py`'s `shutdown_app()` calls `pool.shutdown()` → per-engine
  `shutdown()` — `AsyncLLM.shutdown()` (`async_llm.py:259-272`) already
  handles subprocess teardown; `EnginePool.shutdown()`'s defensive
  `getattr(engine, "shutdown", None)` + exception-swallowing wrapper needs no
  change.
- **Trade-off**: Decision 7's timeout behavior change means a timed-out
  request now consumes fewer engine cycles post-timeout than today, which is
  strictly better, but it is a real behavior change worth flagging to anyone
  benchmarking timeout-adjacent scenarios.
- **Risk**: `_log_kv_cache_stats()` (Decision 6) was already flagged in
  today's code as introspecting non-public, version-specific vLLM internals.
  The migration keeps that same risk profile (still non-public internals),
  just against a different root object.

## Migration Plan

1. Add `AsyncLLM` construction inside `VLLMEngine.__init__`, replacing
   `LLM(...)` + `EngineDriver(...)`.
2. Relocate `derive_terminal_metadata` (Decision 8).
3. Rewrite `generate_stream()` to consume `AsyncLLM.generate()` directly,
   translating `RequestOutput` → `ChatStreamChunk` exactly as today.
4. Rewrite `generate()` to derive from `generate_stream()` (Decision 2).
5. Rewrite `is_healthy()` (Decision 5) and `_log_kv_cache_stats()`
   (Decision 6).
6. Delete `engines/driver.py`, delete `_POOL_STEP_LOCK`, remove the
   `inference_x.engines.driver` mypy baseline entry.
7. Delete `tests/unit/test_engine_driver.py`; add its reproduced
   property-level assertions as new tests against the AsyncLLM-backed engine.
8. Verify `TestReservationLifecycle` (9 tests, `test_chat_service.py`) passes
   unmodified.

## Validation properties (required)

- `ruff check .`, `mypy src/`, `pytest tests/unit` all green.
- No new entry in the mypy baseline for `services.chat_service`,
  `routing.admission`, or `api.deps` (Compatibility Invariant 2 — if any of
  these need a suppression, the boundary was not actually preserved).
- `inference_x.engines.vllm_engine`'s baseline entry may persist (vLLM's
  typing is not fully exposed) or shrink, but must not need a new suppression
  specifically for the `generate`/`generate_stream` override relationship —
  the existing platform-spec requirement "Future generate_stream
  implementations type-check without suppression" applies to this
  implementation.
- New regression test: full stream consumption releases the admission
  reservation exactly once (unchanged); a new test asserting that
  disconnecting mid-stream results in an observable `abort_requests_async`
  call (or equivalent engine-side signal), not just consumer-side queue
  abandonment (Decision 4).
