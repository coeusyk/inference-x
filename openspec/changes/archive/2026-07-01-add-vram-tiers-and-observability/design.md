# Design: add-vram-tiers-and-observability

## Overview
No new top-level module boundary is introduced for sizing or tiers — `vram_tiers.py`
joins `utils/`, alongside the existing `vllm_pool_config.py` it extends. Observability
fills in stub files inside the already-established `observability/` and `api/routes/`
boundaries. The vertical slice: probed VRAM → resolved tier (logged) → quant-aware
per-model sizing (unchanged call sites, new bytes/param lookup) → engine load → live
metrics surfaced over HTTP.

## Module structure

```
src/inference_x/utils/
├── vllm_pool_config.py   # existing; _bytes_per_param() lookup added, threaded through
│                          # estimate_weight_gib/estimate_engine_footprint_gib/
│                          # validate_pool_fits/pool + sequential + single-engine sizing
└── vram_tiers.py          # new; VramTier dataclass, load_tiers(), resolve_tier()

config/
├── vram_tiers.yaml         # new; 6gb/12gb/24gb tier table (declarative, comments carry
│                            # the reasoning per tier — see file for the full contract)
└── models.yaml             # +qwen2.5-7b-awq (quantization: awq, sized for 12GB tier)

src/inference_x/core/settings.py
└── AppSettings.get_vram_tier()   # profile_hardware() -> resolve_tier(); local import
                                   # to keep benchmarks.hardware out of lightweight
                                   # settings consumers/tests

src/inference_x/api/deps.py
└── initialize_app()   # resolves + logs the tier at startup; failure is non-fatal
                        # (warning only) since tiers are advisory in this phase

src/inference_x/schemas/metrics.py    # new content; was 0 bytes
├── ModelVramBreakdown   # name, quantization, estimated_weights_gib,
│                        # kv_capacity_tokens (real, post-load), max_model_len
├── VramSummary          # total_gib, free_gib, models: list[ModelVramBreakdown]
└── MetricsResponse      # request metrics + vram: VramSummary

src/inference_x/api/routes/metrics.py   # new content; was 0 bytes
└── GET /v1/metrics   # MetricsService.summary() + per-loaded-model VRAM breakdown

src/inference_x/engines/vllm_engine.py
└── VLLMEngine.kv_capacity_tokens   # property; num_gpu_blocks * block_size, read from
                                     # llm_engine.vllm_config.cache_config (vLLM 0.22.1's
                                     # V1 LLMEngine path — cache_config is NOT a direct
                                     # llm_engine attribute on this version)

src/inference_x/observability/
├── middleware.py   # _wrap_and_record_sse(): wraps body_iterator for streaming chat
│                    # 200s; measures TTFT at first chunk, approximates completion
│                    # tokens via whitespace word count over delta.content per SSE
│                    # line, records exactly one RequestRecord on stream end
├── recorder.py      # MetricsRecorder.record() gains ttft_ms/tokens_per_sec params
└── storage.py        # RequestRecord gains ttft_ms/tokens_per_sec fields (both None
                       # for non-streaming requests)

src/inference_x/services/metrics_service.py
└── MetricsSummary   # +avg_ttft_ms/avg_tokens_per_sec, averaged only over records
                      # that have them (streaming requests) — a mix of streaming and
                      # non-streaming traffic doesn't skew the mean toward None-as-zero

src/inference_x/schemas/model.py
└── ModelObject   # +quantization, +max_model_len, +estimated_weights_gib (additive)
```

## Key decisions

**Quant-aware bytes/param is a lookup table, not a formula.** AWQ/GPTQ/int4 land at
~0.55 bytes/param (not a bare 0.5) to account for group-wise scale/zero-point overhead;
int8/fp8 at 1.0. Substring fallback (`"gptq-4bit"` matches `"gptq"`) handles variant
spellings without a config-schema change. Unknown quantization strings fall back to bf16
(2 bytes/param) — the conservative direction (over-reserve, don't under-reserve and OOM).

**Tiers are resolved and logged, not enforced, in this phase.** `vram_tiers.yaml`'s
`gpu_memory_utilization_ceiling`, `max_num_seqs`, `block_size`, and `kv_cache_dtype`
fields are declared now (so the tier contract is complete on disk) but not yet consumed
by the engine or an admission layer — that's explicit Phase 2 scope (`docs/DECISIONS.md`
DEC-037). `resolve_tier()` never guesses a higher tier than the evidence supports: a GPU
below every tier's floor still gets the lowest (most conservative) tier, with a warning.

**SSE observability wraps the iterator instead of buffering.** Buffering would defeat
streaming. The wrapper passes each chunk through unmodified and only inspects it for
`data: {...}` lines to accumulate an approximate token count — the same word-count
approximation the playground and `benchmarks/runner.py` already use, since vLLM's
streaming response never carries a final `usage` block (DEC-023). Exactly one
`RequestRecord` is emitted per stream (on normal end or `GeneratorExit` from a client
disconnect); the middleware's unconditional `record()` call is skipped on this branch to
avoid double-counting. A known gap: a mid-stream *engine* failure isn't observable as
`error=True` here, because Starlette's `BaseHTTPMiddleware` only surfaces the inner
task's exception after `dispatch()` has already returned — from the wrapper's point of
view the iterator just ends early, indistinguishable from a clean end-of-stream.

**`kv_capacity_tokens` reads a version-specific vLLM attribute path.** vLLM 0.22.1's V1
`LLMEngine` keeps `CacheConfig` under `llm_engine.vllm_config.cache_config`, not as a
direct `llm_engine.cache_config` attribute (that reads `None` on this version) — verified
by introspecting a loaded engine. Regression-tested in `test_vllm_engine_stream.py`
against the `VLLMEngine.__new__` + MagicMock harness the file already uses.
