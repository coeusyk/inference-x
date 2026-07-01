## 1. Tier and model schema

- [x] 1.1 `config/vram_tiers.yaml`: add `max_num_batched_tokens`,
  `enable_prefix_caching` per tier (6gb: 2048 / false; 12gb: 4096 / true; 24gb:
  8192 / true)
- [x] 1.2 `utils/vram_tiers.py`: extend `VramTier` dataclass + `from_dict` with the
  two new fields, `.get()`-defaulted for backward compatibility
- [x] 1.3 `schemas/model.py`: add `ModelEntry.max_num_batched_tokens: Optional[int]`
  override (same `Field(ge=1, description=...)` shape as `max_num_seqs`)

## 2. Knob resolution

- [x] 2.1 New `apply_tier_knobs(model_config, tier)` in `utils/vllm_pool_config.py`
  resolving `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching` via
  `min(model override or tier value, tier value)` (tier-only for
  `enable_prefix_caching`)
- [x] 2.2 Unit tests: model override smaller than tier value honored; larger value
  clamped down to tier ceiling; no-tier fallback behaves like today

## 3. Engine wiring

- [x] 3.1 `VLLMEngine.__init__`: pass `max_num_batched_tokens`,
  `enable_prefix_caching`, `block_size`, `kv_cache_dtype` into `LLM(**kwargs)` when
  present, defensively tolerant of vLLM version differences
- [x] 3.2 `api/deps.py`: `_build_engine_pool` resolves the VRAM tier (fail-open with
  a warning, same pattern as `_build_admission_controller`) and calls
  `apply_tier_knobs` per model before constructing each `VLLMEngine`

## 4. Admission sequence-count gate

- [x] 4.1 `routing/admission.py`: new `_InFlightSeqTracker` (per-model in-flight
  request count, thread-safe, same shape as `_KVReservationTracker`)
- [x] 4.2 `AdmissionController.admit()`/`release()`: increment/decrement the
  tracker; reject with `EngineSaturatedError` (429) for both priorities when the
  in-flight count is at or above the resolved `max_num_seqs` — no clamp path
- [x] 4.3 Unit tests: saturation at `max_num_seqs` rejects both interactive and
  batch requests; release() frees a slot; no-tier fallback skips this gate
  (fail-open, consistent with the existing KV gate's posture)

## 5. Validation

- [x] 5.1 `uv run pytest tests/unit -v` — full suite passes, including existing
  371+ tests unmodified
- [x] 5.2 Live check: loading `qwen2.5-7b-awq` standalone with the 6gb tier's
  resolved knobs does not OOM at startup (or, if a GPU isn't available for this
  round, a unit test asserting the exact resolved kwargs dict for representative
  tier × model-override combinations)

## 6. Docs and decisions

- [x] 6.1 Add a DEC entry to `docs/DECISIONS.md` documenting the knob wiring and
  the new admission gate
- [x] 6.2 Archive this change to `openspec/changes/archive/` when complete
