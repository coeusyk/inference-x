# Design — add-deterministic-execution (C3 + C4b)

## Context

Authoritative sources: REVIEW §9-C3; Phase C plan §4.3 / §9 / §11; DEC-052 (`strict` must not absorb
determinism). C1 already reserved `ManifestRuntime.batch_invariant` and documents that C3 owns the
wiring. C4a introduced the gated GPU test tier this change's conformance suite reuses.

OPEN #6 verification (this session, against pinned `vllm==0.22.1` on an SM 8.9 host):

| Signal | Result |
|---|---|
| `VLLM_BATCH_INVARIANT` env | Present (`vllm.envs`) |
| `init_batch_invariance()` / `enable_batch_invariant_mode()` | Present |
| `VLLM_DETERMINISM_WARMUP_ITERATIONS` | **Absent** in 0.22.1 |
| Local GPU SM | 8.9 (≥ 8.0) |

## Goals / Non-Goals

**Goals**
- First-class `deterministic` request + startup flag wiring batch-invariant mode on SM ≥ 8.0.
- Hard refuse (400 / boot-fail) on unsupported hardware when determinism is demanded.
- Honest `runtime.batch_invariant` on the run manifest.
- `make verify-determinism`: 1 unique sample over N gated-GPU trials.
- Docs state only the scoped claim.

**Non-Goals**
- End-to-end / cross-hardware / cross-version reproducibility claims.
- Folding into `strict` (DEC-052).
- C2 co-batch identity extraction.
- Bumping vLLM solely to chase a non-existent warmup env var name.

## Decisions

### D1 — `deterministic` is a new field; never an alias of `strict`
Follows DEC-052. Request schema gains `deterministic: bool = False`. Startup config
(`INFERENCE_X_DETERMINISTIC=1` or settings equivalent) enables process-wide mode before engine
construction. `strict` continues to mean only substitution→rejection.

### D2 — Wire `VLLM_BATCH_INVARIANT` before `AsyncLLM` construction; call vLLM init hooks
When deterministic mode is on and SM ≥ 8.0: set `os.environ["VLLM_BATCH_INVARIANT"]="1"` early
enough that `vllm.envs` / `init_batch_invariance()` see it, then construct `VLLMEngine` as today.
Prefer calling `vllm.model_executor.layers.batch_invariant.init_batch_invariance()` explicitly after
env set if import order would otherwise miss it. Record `runtime.batch_invariant=True` only when
the env is actually set and init succeeded.

### D3 — App-level warmup replaces the missing `VLLM_DETERMINISM_WARMUP_ITERATIONS`
vLLM 0.22.1 has no such env var (OPEN #6 partial). When deterministic mode enables successfully,
run **N=3** short warm generate calls (fixed tiny prompt, `max_tokens=1`, temperature 0) on the
live engine before serving — an Inference-X responsibility, not a fabricated vLLM env. Document
the pin (`vllm==0.22.1`) so a future vLLM-native warmup can replace this without changing the
external contract.

### D4 — SM probe + hard refuse
Add `utils/cuda_env.py` (or adjacent) `probe_compute_capability() -> tuple[int,int] | None`.
Supported iff major > 8 or (major == 8 and minor >= 0), i.e. SM ≥ 8.0. Unsupported + deterministic
demanded → startup `RuntimeError` / process exit for the startup flag; HTTP **400** with a stable
error code for per-request `deterministic: true` (plan §4.3 REC: refuse rather than lie).

### D5 — Scoped guarantee language
Docs and error messages claim: under `deterministic: true` on SM ≥ 8.0 with batch-invariant mode
active, **repeated identical single-request trials on the same process/hardware/vLLM version** are
expected to match (`make verify-determinism`). No claim of concurrent multi-request bitwise
identity beyond what vLLM's batch-invariant mode provides, and no cross-machine guarantee.

### D6 — Conformance suite (C4b) lives under `tests/oracle/`, gated like C4a
`tests/oracle/test_determinism.py` + `make verify-determinism`. Same
`INFERENCE_X_RUN_GPU_TESTS=1` + CUDA skip guard. Pass: exactly 1 unique completion text over N
trials (default N=5) with fixed seed + `deterministic: true`.

### D7 — Per-request `deterministic: true` refuses unless startup already activated it (post-merge correction)
Adversarial review of the merged implementation (2026-08-11) found that the original per-request
path called `ensure_deterministic_mode()` directly against the already-running engine. vLLM may
capture CUDA graphs during `AsyncLLM.from_engine_args(...)` (default `enforce_eager=False`); a
graph captured before the batch-invariant dispatcher override is installed keeps replaying the
original kernels no matter what the env var says afterward, so a late per-request activation could
report `runtime.batch_invariant: true` on a run that didn't actually get invariant kernels for
graph-covered shapes — the exact "silently run non-deterministically" failure D4 exists to prevent,
just on the per-request path instead of the SM-unsupported path D4 originally named. Corrected to
D4's own principle: per-request `deterministic: true` now refuses (400,
`code: "deterministic_unsupported"`) unless the process was started with deterministic mode enabled
(`INFERENCE_X_DETERMINISTIC=1`). `ensure_deterministic_mode()` is now only ever invoked at startup
(before engine construction, per D2) and no longer swallows a genuine `init_batch_invariance()`
failure — both call sites now fail closed rather than reporting success the manifest can't back up.

## Risks / Trade-offs

- **[Risk] App-level warmup may be insufficient vs a future vLLM-native warmup.** → Mitigation: D3
  documents the pin; conformance suite is the acceptance gate; revisit on vLLM upgrade.
- **[Risk] SM 8.0 family behavior differs (Ampere triton overrides vs Hopper cublas path).** →
  Mitigation: rely on vLLM's `enable_batch_invariant_mode()` branching; refuse below 8.0 only.
- **[Risk] Operators set `VLLM_BATCH_INVARIANT` manually without our flag.** → Mitigation: C1
  already reads the env honestly into the manifest; C3 does not force it off.

## Open Questions

None that block implementation after D3's resolution of the missing warmup env. Residual product
question (single-request vs full concurrent batch identity) is answered as **scoped single-request
conformance plus vLLM's batch-invariant mode** (D5) — not left open.
