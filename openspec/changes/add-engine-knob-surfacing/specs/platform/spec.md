# Platform Specification Delta — add-engine-knob-surfacing

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

#### Scenario: Sequence-concurrency ceiling is enforced pre-dispatch
- WHEN the number of in-flight requests against a model is at or above its
  resolved `max_num_seqs`
- THEN a new request for that model is rejected with HTTP 429 and a
  `Retry-After` header, regardless of request priority
- AND the engine itself is never invoked for the rejected request

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
