## 1. Schema and registry

- [x] 1.1 `schemas/model.py`: add `ModelEntry.family: Optional[str]`
- [x] 1.2 `services/model_service.py`: add `ModelRegistry.variants(family) ->
  list[ModelEntry]` (groups by `family`, falls back to `name` for ungrouped
  entries)
- [x] 1.3 Unit tests: ungrouped entries behave identically to today; grouped
  entries return all family members

## 2. Variant selector

- [x] 2.1 New `src/inference_x/routing/variant_selector.py`:
  `select_variant(family, registry, tier, available_vram_gib) -> str`,
  `NoVariantFitsError`
- [x] 2.2 Sort variants by descending `_QUANT_BYTES_PER_PARAM[quantization]` (reuse
  from `utils/vllm_pool_config.py`, no new rank field)
- [x] 2.3 Return first variant whose estimated weight size fits
  `available_vram_gib * tier.gpu_memory_utilization_ceiling`
- [x] 2.4 Raise `NoVariantFitsError` naming the family and each variant's estimated
  size vs. budget when none fit
- [x] 2.5 Unit tests: highest-precision-that-fits is chosen; fallback to
  lower-precision when higher doesn't fit; no-fit raises with a clear message;
  single-variant family returns that variant unconditionally

## 3. Deps wiring

- [x] 3.1 `api/deps.py`: before existing pool-fit logic, resolve each name in
  `loaded_models` — if it matches a concrete `ModelEntry.name`, use it as-is
  (today's behavior); otherwise treat it as a family name and resolve via
  `select_variant`
- [x] 3.2 Unit tests: concrete name bypasses the selector; family name resolves to
  a concrete variant; unresolvable name (neither a concrete entry nor a known
  family) raises the existing "model not registered" error unchanged

## 4. Example config

- [x] 4.1 `config/models.yaml`: add `family:` to a representative set of variants
  (e.g. a qwen2.5-7b bf16/int8/awq trio) as a worked example — no changes to
  existing ungrouped entries

## 5. Validation

- [x] 5.1 `uv run pytest tests/unit -v` — full suite passes, including existing
  tests for `INFERENCE_X_LOADED_MODELS` with concrete names unmodified
- [x] 5.2 New `tests/unit/test_variant_selector.py` covers all selector unit tests
  from task 2.5

## 6. Docs and decisions

- [x] 6.1 Add a DEC entry to `docs/DECISIONS.md` documenting variant grouping and
  the load-time selection algorithm
- [x] 6.2 Archive this change to `openspec/changes/archive/` when complete
