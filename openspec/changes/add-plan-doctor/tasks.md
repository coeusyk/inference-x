# Tasks — add-plan-doctor (C6)

> Implementation checklist for the approval-gated build phase. Drafting this list is not
> implementation; do not check items until the corresponding code lands.

## 1. Plan endpoint
- [x] 1.1 Add `api/routes/plan.py`: `GET /v1/plan`, mirroring `models.py`'s route shape.
- [x] 1.2 Response schema (`schemas/`): one entry per registered model — `estimated_weight_gib`,
      `estimated_kv_cache_gib`, `estimated_footprint_gib`, `gpu_memory_utilization`, resolved tier
      knobs (`block_size`, `kv_cache_dtype`, `enable_prefix_caching`, `max_num_seqs`,
      `max_num_batched_tokens`).
- [x] 1.3 Compute via `utils/vllm_pool_config.py`'s existing functions and `vram_tiers.resolve_tier`
      — no new sizing arithmetic (D2).
- [x] 1.4 Unit tests: per-model entries present; values match direct calls to the same
      `vllm_pool_config` functions; no engine construction occurs during the request.

## 2. Doctor endpoint
- [x] 2.1 Add `api/routes/doctor.py`: `GET /v1/doctor`, mirroring `health.py`'s route shape.
- [x] 2.2 Environment section from `benchmarks/hardware.py::profile_hardware()` (D4) plus
      `utils/vllm_pool_config.py::probe_gpu_memory_gib()`.
- [x] 2.3 Per-model fit section: catch `validate_model_fits`'s `ValueError` per model, report
      `{fits: false, reason: <message>}` on failure or `{fits: true}` on success (D3).
- [x] 2.4 Unit tests: a model sized to fit reports `fits: true`; a model sized to exceed probed
      VRAM reports `fits: false` with the underlying reason string; no engine construction occurs.

## 3. CLI wrappers
- [x] 3.1 `make plan` / `make doctor` targets in `Makefile`, calling the endpoints (server must be
      running, consistent with the existing `make benchmark` convention) and pretty-printing.
- [x] 3.2 Update `make help`'s endpoint list to include `GET /v1/plan` and `GET /v1/doctor`.

## 4. Validation & docs
- [x] 4.1 `openspec validate add-plan-doctor --strict` passes.
- [x] 4.2 Unit suite, `ruff`, `mypy` green (mockable tests only; no GPU path required — every
      function this change calls is already GPU-independent per D5). 602 passed, 1 pre-existing
      unrelated xfail; ruff clean; mypy clean, 54 files. Also live-verified against real
      `config/models.yaml` and real hardware (RTX 4060/WSL2): `/v1/plan` reported all 8 registered
      models with sizing matching the engine construction path's own log lines; `/v1/doctor`
      correctly reported `qwen2.5-7b-bf16` as not fitting (7.3 GiB free vs ~13.0 GiB needed) with
      the exact `validate_model_fits` reason, and every other (smaller/quantized) model as fitting.
