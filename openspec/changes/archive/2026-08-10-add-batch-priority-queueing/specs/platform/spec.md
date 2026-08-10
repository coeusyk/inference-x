<!--
This delta is written against the "Incremental architecture" requirement's
target end-state after `rescope-admission-control` (B4) archives — i.e. it
starts from B4's own pending MODIFIED text (bounded-wait-then-429), not from
the currently-committed `specs/platform/spec.md`, which still describes B4's
now-superseded instant-rejection wording. B4 is merged to `develop` and its
bounded-wait behavior is live in `admission.py` today; only its OpenSpec
archive step is still pending. This change's own archive is expected to
happen after B4's (`tasks.md` §6.3), so building on B4's target text rather
than the stale committed text avoids describing a scenario that never
matches what ships.

The archiver convention (a MODIFIED requirement carries the entire existing
block) is followed here against that target text: the "Tier and model
knobs," "Knob resolution is unavailable," and "Change is applied" scenarios
are pasted unchanged from B4's own delta. Only "Sequence-concurrency
ceiling" is further modified — its "regardless of request priority" clause
is no longer accurate once wait duration is priority-differentiated — and
one new scenario is added for the batch-queue-depth backpressure this
change introduces (design.md §3).

`tasks.md` §1-5 are complete — implementation and tests exist on
`feat/batch-priority-queueing`. §6 (documentation/archive) has not started.
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

#### Scenario: Sequence-concurrency ceiling is enforced with a priority-differentiated bounded wait
- WHEN the number of in-flight requests against a model is at or above its
  resolved `max_num_seqs`
- THEN a new `interactive`-priority request waits, up to a short, bounded,
  configurable deadline, for an in-flight request to complete and free a slot
- AND a new `batch`-priority request waits, up to a separate and
  substantially longer bounded, configurable deadline, for a slot to free
- AND if a slot frees within the request's applicable deadline, the request
  is admitted and proceeds exactly as it would have if the ceiling had never
  been hit
- AND if no slot frees before the applicable deadline elapses, the request
  is rejected with HTTP 429 and a `Retry-After` header
- AND the engine itself is never invoked for a request that is ultimately
  rejected after its wait elapses

#### Scenario: Batch-priority queue depth is bounded
- WHEN a `batch`-priority request arrives for a model whose count of
  currently-waiting `batch`-priority requests is already at that model's
  configured queue-depth capacity
- THEN the new request is rejected immediately with HTTP 429 and a
  `Retry-After` header, without waiting and without ever counting toward or
  affecting the sequence-concurrency wait
- AND `interactive`-priority requests are never subject to this capacity
  check

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
