# Proposal: add-deterministic-execution (Phase C — C3 + C4b)

## Why

`docs/REVIEW-2026-08-03-architecture.md` §9-C3 and the locked Phase C plan §4.3: clients need a
**scoped** determinism contract — `deterministic: true` wiring vLLM's batch-invariant mode on
supported GPUs (SM ≥ 8.0), refusing rather than silently running non-deterministically on
unsupported hardware, recording `runtime.batch_invariant` on the C1 manifest, and shipping the
conformance check (`make verify-determinism`: 1 unique sample over N trials) in the **same**
change (C4b). DEC-052 forbids folding this into `strict`.

OPEN #6 (vLLM support verification) is resolved for the pinned `vllm==0.22.1`: `VLLM_BATCH_INVARIANT`
and `init_batch_invariance()` / `enable_batch_invariant_mode()` exist. The plan's named
`VLLM_DETERMINISM_WARMUP_ITERATIONS` env var does **not** exist in 0.22.1; this change uses an
application-level warmup instead (see design.md) rather than blocking on a speculative vLLM bump.

## What Changes

- **Request + startup `deterministic` flag.** New first-class boolean (not `strict`). When true on
  supported hardware: set `VLLM_BATCH_INVARIANT=1` before engine construction / call vLLM's
  batch-invariance init, run a small app-level warmup, and record `runtime.batch_invariant: true`.
- **Hard refuse on unsupported hardware.** SM &lt; 8.0 (or no CUDA): startup with deterministic
  enabled fails fast; a per-request `deterministic: true` returns HTTP 400 rather than degrading.
- **`make verify-determinism` (C4b).** Gated GPU suite: N trials of the same seeded request must
  produce exactly 1 unique sample. Reuses the C4a GPU opt-in (`INFERENCE_X_RUN_GPU_TESTS=1`).
- **Docs.** Scoped claim only — same hardware + same vLLM version; never "end-to-end reproducible."

## Capabilities

### Modified Capabilities

- `platform`: ADDS determinism contract requirements (flag, refuse-on-unsupported, manifest
  `batch_invariant`, gated verify-determinism). May MODIFY documentation / comparability wording
  only where needed to state the scoped claim.

### New Capabilities

- (none)

## Impact

- **Additive request field** with default `false` — existing clients unchanged.
- **Affected code:** `schemas/chat.py`, engine / deps startup path, SM probe helper, error mapping
  (400), manifest assembly (`ManifestRuntime.batch_invariant` already reserved by C1),
  `tests/oracle/` or `tests/oracle/`-adjacent determinism suite, `Makefile`, brief docs.
- **Out of scope:** C2 co-batch extraction; folding into `strict` (DEC-052); claiming cross-hardware
  or cross-version reproducibility; Varex e2e (C5).
