# Design — add-plan-doctor (C6)

## Context

Authoritative source: `docs/REVIEW-2026-08-03-architecture.md` §6.2c ("the read-only preflight
triad"). Planning report: `docs/PHASE-C-EXECUTION-PLAN.md` §4.7, §9 — C6 is independent of every
other Phase C change; it consumes C1's (`add-run-manifest`) hardware sub-schema for environment
reporting rather than re-deriving hardware detection, but does not depend on C1's `run_id`/manifest
machinery itself.

## Goals / Non-Goals

**Goals**
- `GET /v1/plan` — per-model sizing report, computed from `utils/vllm_pool_config.py`'s existing
  functions, never re-implemented.
- `GET /v1/doctor` — read-only readiness report: probed GPU memory, hardware profile, per-model fit.
- `make plan` / `make doctor` CLI wrappers over the same computation.

**Non-Goals (deferred)**
- `tune` (measure-and-save a machine-local execution profile) — a separate, later roadmap item.
- Any endpoint that starts an engine, or that changes `vllm_pool_config.py`'s sizing arithmetic.
- Auth/access control — no route in this API is authenticated today (local dev server); `plan` and
  `doctor` follow that existing convention rather than introducing a new one.

## Decisions

### D1 — `plan` reports every registered model, not only currently-loaded ones
`ModelRegistry.all()` lists every model declared in `config/models.yaml`, independent of
`INFERENCE_X_LOADED_MODELS`. `plan`'s value is answering "what would loading X cost" *before*
committing to load it — reporting only already-loaded models would make it redundant with
`GET /v1/models`' existing `estimated_weights_gib` field. Sizing (`gpu_memory_utilization`, tier
knobs) is computed against the VRAM tier the *current machine* resolves to
(`vram_tiers.resolve_tier`, the same call `api/deps.py` already makes at startup) — `plan` describes
"if this model were loaded on this box right now," not a hypothetical different GPU.

### D2 — `plan`'s footprint numbers are the same functions the engine startup path uses, called directly
`estimate_weight_gib`, `estimate_kv_cache_gib`, `estimate_engine_footprint_gib`, `apply_tier_knobs`,
and `scale_model_config` are imported and called as-is from `utils/vllm_pool_config.py` — the route
handler is a thin aggregator, not a second sizing implementation. This is the same discipline C1's
manifest assembly used for `resolved`/`warnings`/`timing` (design.md D-none, but tasks.md 2.2):
"no re-derivation of existing signals."

### D3 — `doctor`'s per-model fit check reuses `validate_model_fits` and reports its exact failure reason
`validate_model_fits` already raises `ValueError` with a specific, actionable message (e.g. "Only
X GiB VRAM free but model Y needs ~Z GiB for weights..."). `doctor` catches that exception per model
and reports `{"model": name, "fits": false, "reason": "<the ValueError message>"}` rather than a
generic boolean — the message already exists and is more useful than reinventing a shorter one.
A model that fits reports `{"model": name, "fits": true}`.

### D4 — `doctor`'s environment section reuses C1's hardware profile function, not a new probe
`benchmarks/hardware.py::profile_hardware()` (already the source of C1's manifest `hardware` block)
supplies `doctor`'s environment section verbatim: GPU name, driver, CUDA version, WSL2 flag, CPU
name, total/free VRAM. No second hardware-detection code path is introduced. This is the concrete
form of "C6 consumes C1's hardware sub-schema" from the planning report's dependency graph.

### D5 — `doctor` never starts an engine to validate its report
Every check `doctor` performs (`probe_gpu_memory_gib`, `validate_model_fits`, `profile_hardware`)
is already engine-independent — none of them requires a running vLLM process. `doctor` therefore
answers "would this model load" without the cost or side effects of actually loading it, preserving
§6.2c's "read-only preflight" framing. If a model's `doctor` report says `fits: true` but the engine
still fails to construct at startup, that is a `doctor`/reality drift to fix in `doctor`'s checks,
not a reason to make `doctor` load engines.

### D6 — [OPEN] Should `plan`/`doctor` accept a `model` query filter?
Reporting every registered model on every call is simplest and matches `GET /v1/models`' existing
shape (no filter param). A large model registry could make the response large, but Inference-X's
registries in practice are single-digit model counts. Deferred as a non-blocking follow-up rather
than speculatively built now — add a `?model=` filter if a real registry size makes it necessary.

## Risks / Migration

- **Additive and read-only.** No existing endpoint, schema, or engine-construction path changes.
  Both new routes are new files; nothing in `api/deps.py`'s engine-pool construction is touched.
- **No new failure modes.** Both routes catch the same `ValueError` shapes `vllm_pool_config.py`
  already raises and turn them into report fields, not HTTP errors — a model that would fail to
  load is not itself a `doctor` request failure.
