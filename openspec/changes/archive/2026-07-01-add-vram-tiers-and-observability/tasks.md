# Tasks: add-vram-tiers-and-observability

## 1. Quant-aware VRAM sizing
- 1.1 Add `_QUANT_BYTES_PER_PARAM` lookup + `_bytes_per_param()` to
  `utils/vllm_pool_config.py` (awq/gptq/int4 ≈ 0.55, int8/fp8 = 1.0, unknown → bf16 2.0)
- 1.2 Thread `quantization` through `estimate_weight_gib()`,
  `estimate_engine_footprint_gib()`, `_minimum_utilization()`,
  `_weights_only_utilization()`, `_weight_scaled_utilization()`,
  `_apply_sequential_vram_caps()`, `_single_engine_utilization()`, `validate_pool_fits()`
- 1.3 Unit tests: quant-aware weight estimate vs. bf16 baseline for the same param count

## 2. VRAM tiers
- 2.1 Add `config/vram_tiers.yaml` — 6gb/12gb/24gb tiers with
  `gpu_memory_utilization_ceiling`, `max_model_len_cap`, `max_num_seqs`, `block_size`,
  `kv_cache_dtype`, and comments documenting the reasoning per tier
- 2.2 Add `utils/vram_tiers.py` — `VramTier` dataclass, `load_tiers()` (cached, sorted
  ascending by `min_vram_gb`), `resolve_tier()` (highest tier cleared; falls back to
  lowest with a warning if below every floor)
- 2.3 Add `AppSettings.get_vram_tier()` (`core/settings.py`) — local import of
  `benchmarks.hardware`/`utils.vram_tiers` to avoid pulling them into lightweight
  settings consumers
- 2.4 Wire tier resolution + logging into `deps.initialize_app()` (non-fatal on failure)
- 2.5 Unit tests: `load_tiers`/`resolve_tier` boundary conditions, missing/empty config,
  fallback-to-lowest-tier warning path, real shipped `vram_tiers.yaml` round-trip

## 3. Quantized model entry
- 3.1 Add `qwen2.5-7b-awq` (`quantization: awq`) to `config/models.yaml`, sized for the
  12GB tier; document why it's excluded from the default 6GB `INFERENCE_X_LOADED_MODELS`
- 3.2 Unit test coverage in `test_model_registry.py` for the new entry

## 4. Observability HTTP surface
- 4.1 Fill `schemas/metrics.py` (`MetricsResponse`, `VramSummary`, `ModelVramBreakdown`)
- 4.2 Fill `api/routes/metrics.py` (`GET /v1/metrics`); register in `api/main.py`
- 4.3 Add `get_engine_pool()` / `get_metrics_service()` deps in `api/deps.py`
- 4.4 Expose `VLLMEngine.kv_capacity_tokens` (real post-load KV capacity) and
  `model_name`/`model_path` properties
- 4.5 Unit tests for the route (models/registry/pool fakes) and schema round-trip

## 5. Streaming TTFT / tokens-per-sec
- 5.1 Add `_wrap_and_record_sse()` + `_count_sse_delta_tokens()` to
  `observability/middleware.py`; wrap `body_iterator` for streaming chat 200s instead of
  skipping token extraction
- 5.2 Add `ttft_ms`/`tokens_per_sec` fields to `RequestRecord` (`storage.py`) and
  `MetricsRecorder.record()` (`recorder.py`)
- 5.3 Add `avg_ttft_ms`/`avg_tokens_per_sec` to `MetricsSummary`
  (`services/metrics_service.py`), averaged only over records that have them
- 5.4 Unit tests: SSE wrapper TTFT/token accounting, client-disconnect
  (`GeneratorExit`) path, non-double-counting against the middleware's normal
  `record()` call

## 6. Model metadata
- 6.1 Add `quantization`, `max_model_len`, `estimated_weights_gib` to `ModelObject`
  (`schemas/model.py`) and `GET /v1/models` (`api/routes/models.py`)
- 6.2 Unit tests for the extended response shape

## 7. Validation
- 7.1 `uv run pytest tests/unit -v` — 352/352 pass
- 7.2 Live smoke: start the server on the 6GB dev box, confirm VRAM tier (`6gb`) resolves
  and logs at startup
- 7.3 Live smoke: `GET /v1/models` returns quantization/VRAM metadata for all six
  registered models including `qwen2.5-7b-awq`
- 7.4 Live smoke: `GET /v1/metrics` returns a live weights/KV/free VRAM breakdown
- 7.5 Live smoke: a streaming chat request populates `avg_ttft_ms`/`avg_tokens_per_sec`
  on the next `GET /v1/metrics` call, with the SSE response body unchanged

## 8. Docs and decisions
- 8.1 Add DEC-037 to `docs/DECISIONS.md`
- 8.2 Add Phase 7 to `docs/PHASES.md` with exit criteria
- 8.3 Archive this change to `openspec/changes/archive/` when complete
