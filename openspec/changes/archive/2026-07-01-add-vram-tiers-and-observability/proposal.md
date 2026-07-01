# Proposal: add-vram-tiers-and-observability

## Summary
Make VRAM budgeting quant-aware, add an explicit per-GPU-class VRAM tier contract, and
wire the previously-stubbed `/v1/metrics` observability surface — including streaming
(SSE) TTFT and tokens/sec, which the middleware could not see before this change.

## Why
`utils/vllm_pool_config.py` sized every model as unquantized bf16
(`_BYTES_PER_PARAM = 2` hardcoded), so a 4-bit (AWQ/GPTQ) model would be sized as if it
needed 3-4x its actual VRAM — either rejected outright or loaded with far too much
headroom reserved. There was no documented, explicit VRAM-capacity contract per class of
GPU (6GB dev box vs. 12GB/24GB hardware), only ad hoc `gpu_memory_utilization: auto`
sizing. Separately, `api/routes/metrics.py` and `schemas/metrics.py` were 0-byte files —
`services/metrics_service.py` existed but had no HTTP surface — and
`ObservabilityMiddleware` skipped token extraction entirely for streaming chat
completions, so TTFT and tokens/sec were invisible for the request shape the playground
and most real traffic actually use.

This is Phase 1 of a larger VRAM-aware architecture plan (see `docs/DECISIONS.md`
DEC-037). Phase 2 (admission control, `max_context_tokens`/`precision`/`priority`
request fields, non-streaming continuous-batching fix) and Phase 3 (CPU offload, prefix
caching) are deferred to future changes.

## In scope
- Quant-aware weight/footprint estimation in `utils/vllm_pool_config.py`, threaded
  through every existing caller (pool scaling, sequential-load caps, single-engine
  sizing, `validate_pool_fits`)
- `config/vram_tiers.yaml` + `utils/vram_tiers.py`: declarative 6gb/12gb/24gb tier table,
  resolved from probed VRAM via `AppSettings.get_vram_tier()` and logged at startup
- One real 4-bit model (`qwen2.5-7b-awq`, AWQ) added to `config/models.yaml` to exercise
  the quantized sizing path
- `GET /v1/metrics`: request metrics (latency/TTFT/tokens-per-sec) plus a live per-model
  VRAM breakdown (weights estimate, real post-load KV capacity, free/total VRAM)
- Streaming chat completions record TTFT and approximate tokens/sec via a wrapped
  `body_iterator` in `ObservabilityMiddleware`, without buffering the stream
- `GET /v1/models` additively exposes `quantization`, `max_model_len`,
  `estimated_weights_gib`

## Out of scope
- Tier *enforcement* (admission control against `max_num_seqs`/`max_model_len_cap`) —
  this change only resolves and logs the tier
- `AdmissionController`, per-request KV pricing, `max_context_tokens`/`precision`/
  `priority` request fields
- Non-streaming continuous-batching fix (`_run_completion` still serializes)
- CPU/weight offload, prefix caching, `kv_cache_dtype=fp8`
- Validation on 12GB/24GB hardware (not available; quant-aware sizing is unit-tested
  only, not live-verified above the 6GB dev tier)
