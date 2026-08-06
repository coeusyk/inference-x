## ADDED Requirements

### Requirement: Native engine metrics are exposed as a separate, additive Prometheus endpoint

The system SHALL expose vLLM's own Prometheus stat logger as `GET /metrics`,
in vLLM's native text exposition format, without modifying or being fed by
`GET /v1/metrics`.

#### Scenario: GET /metrics returns vLLM's own Prometheus text exposition
- **WHEN** a client sends `GET /metrics`
- **THEN** the response has a Prometheus text-exposition `Content-Type`
- **AND** the body includes at least one `vllm:`-prefixed metric series
  sourced from vLLM's own `PrometheusStatLogger`
- **AND** no InferenceX code translates, renames, or reinterprets those
  series — the endpoint is a passthrough mount

#### Scenario: /metrics scrapes are excluded from /v1/metrics aggregation
- **WHEN** `GET /metrics` is scraped one or more times
- **THEN** `GET /v1/metrics`'s `total_requests`, `avg_latency_ms`, and
  `p95_latency_ms` are unaffected by those scrapes
- **AND** no `RequestRecord` is created for a `/metrics` request

#### Scenario: pool_size > 1 engines are not silently mislabeled
- **WHEN** the runtime is configured with `pool_size > 1`
- **THEN** a `WARNING` is logged at startup naming the metric-label
  collision risk between multiple `AsyncLLM` instances' default
  `PrometheusStatLogger`s
- **AND** no construction-time validation rejects or alters the
  configuration — `pool_size > 1` remains not-guaranteed-and-not-forbidden

#### Scenario: GET /v1/metrics is unaffected by this change
- **WHEN** this change is applied
- **THEN** `GET /v1/metrics`'s response schema (`MetricsResponse`) and every
  field's computation are byte-for-byte unchanged from before this change
- **AND** no `vllm:*` data feeds any `/v1/metrics` field
