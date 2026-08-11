# Proposal: add-plan-doctor (Phase C — C6)

## Why

`docs/REVIEW-2026-08-03-architecture.md` §6.2c: Inference-X already computes almost all of a
Colibrì-style `plan` inside `utils/vllm_pool_config.py` — weight estimation, KV-cache footprint,
tier-knob resolution, and per-model VRAM fit validation — and then throws it away into log lines.
Nothing in the public API or CLI lets an operator ask "what would loading model X actually cost,
and does it fit on this GPU?" without starting the engine and reading logs. §6.2c calls this pair
("the read-only preflight triad", minus `tune`, which is out of scope here) "nearly free" and "the
honest answer to what Inference-X does that `vllm serve` doesn't."

Phase C's planning report (`docs/PHASE-C-EXECUTION-PLAN.md`, approved) lists this as **C6**:
independent of every other Phase C change, consuming C1's (`add-run-manifest`) hardware-provenance
sub-schema for its own environment reporting rather than re-deriving hardware detection.

## What Changes

- **`GET /v1/plan`.** For every registered model (`ModelRegistry.all()`), report the sizing
  computation `vllm_pool_config.py` already performs when an engine actually starts: estimated
  weight GiB, estimated KV-cache GiB (at the model's configured `max_model_len`), total engine
  footprint GiB, the resolved `gpu_memory_utilization`, and the resolved tier knobs
  (`block_size`, `kv_cache_dtype`, `enable_prefix_caching`, `max_num_seqs`,
  `max_num_batched_tokens`) for the currently-resolved VRAM tier. Read-only: computes the same
  values the engine would use, does not start or reconfigure anything.
- **`GET /v1/doctor`.** A read-only readiness report: probed GPU memory (total/free,
  `probe_gpu_memory_gib`), the hardware profile already exposed via C1's manifest
  (`benchmarks/hardware.py`), and per-model fit status (`validate_model_fits` outcome — `ok` or
  the exact reason it would not fit, e.g. insufficient free VRAM for weights). Surfaces the
  platform's existing environment/fit checks as one readable report instead of scattered log
  lines and a 500 at engine-construction time.
- **`make plan` / `make doctor`.** Thin CLI wrappers that call the same underlying computation
  (not a second implementation) and pretty-print it, added to `Makefile` alongside the existing
  `make benchmark` / `make advise` targets.

## Impact

- **Additive, read-only, and backward-compatible.** Two new `GET` endpoints and two new `make`
  targets; no existing endpoint, schema, or behavior changes. Neither endpoint starts, stops, or
  reconfigures an engine — they report what the existing sizing/tier/fit functions already compute.
- **Spec:** ADDS two requirements to `platform` (`GET /v1/plan` reports sizing; `GET /v1/doctor`
  reports readiness). No MODIFIED or REMOVED requirements.
- **Affected code (implementation phase, not this proposal):** new `api/routes/plan.py` and
  `api/routes/doctor.py` (mirroring the existing `models.py`/`health.py` route shape), new response
  schemas in `schemas/` (reusing `benchmarks/hardware.py`'s hardware profile shape rather than a
  parallel one), `Makefile`. Reuses `utils/vllm_pool_config.py`'s existing functions and
  `utils/vram_tiers.py`'s `resolve_tier` verbatim — no new sizing or fit logic is introduced.
- **Explicitly out of scope (deferred):** `tune` (measure-and-save a machine-local profile — a
  separate, later roadmap item per §6.2c, not part of C6), any endpoint that *starts* an engine to
  validate plan/doctor output empirically, and any change to `vllm_pool_config.py`'s sizing
  arithmetic itself.
