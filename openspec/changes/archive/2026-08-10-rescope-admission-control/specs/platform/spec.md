<!--
Round 3 of this investigation resolves what round 2 deliberately withheld.
Option C (a bounded per-model asyncio.Semaphore, see design.md — "Decision:
Option C approved") is now the change owner's confirmed direction, and its
lifecycle/keying/timeout design is fully specified (design.md — "Semaphore
lifecycle analysis", "Acquisition, rejection, and timeout semantics"). The
MODIFIED requirement below replaces the existing "Sequence-concurrency
ceiling is enforced pre-dispatch" scenario's instant-rejection text with
bounded-wait-then-429 semantics, worded around the mechanism rather than a
specific numeric deadline — the exact `admission_wait_s` default for model
sizes this investigation's GPU doesn't represent remains an explicitly
provisional, config-overridable value (design.md, same section), not
something this spec text should hardcode. The other three scenarios of the
"Incremental architecture" requirement are pasted unchanged, per this repo's
archiver convention that a MODIFIED requirement must carry the entire
existing block, not just the delta.

This is still investigation/design phase — no `src/`, `tests/`, or
`pyproject.toml` file is touched by this change (tasks.md section 2, "not
started"). This spec delta describes the decided design, to be merged into
`openspec/specs/platform/spec.md` only when the change is archived after
implementation actually lands, per normal OpenSpec convention.
-->

## MODIFIED Requirements

### Requirement: Incremental architecture
The system SHALL support additive growth without requiring large structural
rewrites, and VRAM/concurrency tuning knobs declared in tier or model config SHALL
actually be applied at engine construction time rather than left inert.

#### Scenario: Tier and model knobs are resolved together
- WHEN a model is loaded under a resolved VRAM tier
- THEN the effective `max_num_seqs` and `max_num_batched_tokens` used to construct
  the engine are the tighter of the tier's ceiling and any per-model override
- AND `block_size`, `kv_cache_dtype`, and `enable_prefix_caching` from the resolved
  tier are passed to the underlying inference engine

#### Scenario: Sequence-concurrency ceiling is enforced with a bounded wait
- WHEN the number of in-flight requests against a model is at or above its
  resolved `max_num_seqs`
- THEN a new request for that model waits, up to a bounded, configurable
  deadline, for an in-flight request to complete and free a slot
- AND if a slot frees within that deadline, the new request is admitted and
  proceeds exactly as it would have if the ceiling had never been hit
- AND if no slot frees before the deadline elapses, the request is rejected
  with HTTP 429 and a `Retry-After` header, regardless of request priority
- AND the engine itself is never invoked for a request that is ultimately
  rejected after its wait elapses

#### Scenario: Knob resolution is unavailable
- WHEN no VRAM tier can be resolved for the current hardware
- THEN engine construction and admission control continue using each model's own
  configured values (or built-in defaults), consistent with this codebase's
  existing fail-open posture for advisory signals

#### Scenario: Change is applied
- WHEN this change is applied
- THEN existing `vram_tiers.yaml` and `models.yaml` files without the new fields
  continue to work unchanged
- AND all existing unit tests continue to pass unchanged

## ADDED Requirements

### Requirement: Admission control stays independent of observability surfaces
The system SHALL make every admission decision (context length, KV-pool
pressure, or sequence concurrency) using only state InferenceX tracks directly
or reads from the engine's synchronous interface, and SHALL NOT read from or
depend on `GET /metrics` or `GET /v1/metrics` to decide whether to admit,
clamp, or reject a request.

#### Scenario: Metrics scraping never influences admission
- **WHEN** any admission decision is made for any gate
- **THEN** the decision does not query, scrape, or otherwise read the
  Prometheus registry or the `/v1/metrics` in-memory store
- **AND** `GET /metrics` and `GET /v1/metrics` continue to report
  observability data only, unaffected by admission-control behavior

#### Scenario: Observability endpoints remain unaffected by this change
- **WHEN** this change (in any future implementation phase) is applied
- **THEN** `GET /metrics` (vLLM-native Prometheus registry) and
  `GET /v1/metrics` (HTTP-boundary aggregates) continue to report exactly the
  same data they reported before this change
- **AND** neither endpoint gains a new dependency on `AdmissionController` or
  vice versa
