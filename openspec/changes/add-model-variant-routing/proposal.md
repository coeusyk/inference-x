## Why

`config/models.yaml` currently lists each model as an independent entry — there is
no way to say "these three entries are the same model at different precisions,
pick whichever fits." Operators moving between GPUs of different VRAM sizes must
manually edit `INFERENCE_X_LOADED_MODELS` (and often `models.yaml` itself) to name
the right concrete variant. DEC-038 explicitly deferred a `precision` request field
as "meaningless without variant sets" — this change is that missing prerequisite.

Phase 1's quant-aware weight estimator (`utils/vllm_pool_config.py`'s
`_QUANT_BYTES_PER_PARAM` table plus its weight-size estimation) already has
everything needed to compare variants by size; nothing currently uses it to choose
between multiple registered entries for the same underlying model.

## What Changes

- `ModelEntry` gains an optional `family: Optional[str]` field
  (`schemas/model.py`). Entries with no `family` are unaffected — they behave
  exactly as today (a "family of one").
- New `ModelRegistry.variants(family) -> list[ModelEntry]` helper
  (`services/model_service.py`) grouping entries by `family` (falling back to
  `name` for ungrouped entries).
- New `routing/variant_selector.py`: `select_variant(family, registry, tier,
  available_vram_gib) -> str`, sorting a family's variants by descending
  bytes-per-param (reusing `_QUANT_BYTES_PER_PARAM` — bf16 > int8/fp8 > awq/gptq/
  int4 — no new precision-rank field, so there is exactly one source of truth for
  precision ordering) and returning the highest-precision variant whose estimated
  weight size fits the available VRAM budget. Raises `NoVariantFitsError` if none
  do.
- `_build_engine_pool`/`initialize_app` (`api/deps.py`) resolve any family name in
  `INFERENCE_X_LOADED_MODELS` to a concrete variant via `select_variant` before
  today's existing pool-fit/load logic runs. A concrete variant name still bypasses
  the selector entirely and loads exactly that entry — both meanings of the env var
  keep working.
- This is a **load-time** decision only (which engine to start). `TaskRouter` and
  `AdmissionController` are unaffected — they only ever see the concrete resolved
  model name, exactly as today.

## Capabilities

### New Capabilities
- `model-variant-routing`: grouping model config entries into families and
  selecting the best-fitting variant at load time based on available VRAM.

### Modified Capabilities
(none — `platform`'s existing model-loading requirements are extended, not
changed in a way that alters existing behavior for ungrouped models)

## Impact

- `src/inference_x/schemas/model.py`: one new optional `ModelEntry` field.
- New `src/inference_x/routing/variant_selector.py`.
- `src/inference_x/services/model_service.py`: one new `ModelRegistry` method.
- `src/inference_x/api/deps.py`: family-name resolution before engine construction.
- `config/models.yaml`: `family:` added only to entries an operator wants grouped;
  no changes required to existing entries.
- No change to `TaskRouter`, `AdmissionController`, or any request/response schema.
