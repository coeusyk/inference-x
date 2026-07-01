# Platform Specification Delta — add-vram-tiers-and-observability

## MODIFIED Requirements

**Requirement: Quant-aware VRAM sizing**
The system SHALL estimate model weight VRAM footprint using the model's declared
quantization scheme, not a fixed bf16 assumption.

Scenario: An AWQ 4-bit model is sized
- WHEN `estimate_weight_gib()` is called with `quantization="awq"`
- THEN the estimate uses ~0.55 bytes/param instead of the bf16 default of 2 bytes/param
- AND every sizing call site (pool scaling, sequential-load caps, single-engine sizing,
  `validate_pool_fits`) receives the same quantization-aware estimate

**Requirement: VRAM tier resolution**
The system SHALL resolve the current machine to a declared VRAM tier at startup, based
on probed total VRAM, and SHALL fall back to the most conservative tier (with a warning)
rather than guess a higher tier when probing is inconclusive.

Scenario: Startup on a known GPU class
- WHEN the server starts and probes total VRAM
- THEN it resolves the highest tier in `config/vram_tiers.yaml` whose `min_vram_gb`
  floor the GPU clears
- AND logs the resolved tier name and its parameters

Scenario: Probed VRAM is below every tier floor
- WHEN probed VRAM is below the lowest tier's `min_vram_gb`
- THEN the system uses the lowest tier anyway
- AND logs a warning explaining the fallback

**Requirement: Live VRAM and request metrics**
The system SHALL expose a read-only `GET /v1/metrics` endpoint returning aggregated
request metrics and a live per-model VRAM breakdown.

Scenario: Metrics are requested after some traffic
- WHEN `GET /v1/metrics` is called
- THEN the response includes total_requests, error_count, avg_latency_ms, p95_latency_ms
- AND a `vram` object with total/free GiB and, per loaded model, quantization,
  estimated weight GiB, real KV-pool capacity in tokens, and max_model_len

**Requirement: Streaming request observability**
The system SHALL record time-to-first-token and an approximate tokens-per-second rate
for streaming (SSE) chat completions, without buffering the response body.

Scenario: A streaming chat completion is served
- WHEN a client sends `stream: true` to `/v1/chat/completions`
- THEN the SSE response body is forwarded to the client unmodified, chunk by chunk
- AND exactly one metrics record is stored for the request once the stream ends
- AND `GET /v1/metrics` reflects the observed TTFT and tokens/sec in its running average

**Requirement: Non-invasive integration**
This change SHALL NOT alter the request/response shape of any existing endpoint beyond
strictly additive fields, and SHALL NOT change Phase 1 API behavior.

Scenario: Change is applied
- WHEN this change is applied
- THEN all Phase 1–6 tests continue to pass unchanged
- AND `GET /v1/models` gains only additive fields (`quantization`, `max_model_len`,
  `estimated_weights_gib`)
- AND `GET /v1/metrics` is a new, additive route
