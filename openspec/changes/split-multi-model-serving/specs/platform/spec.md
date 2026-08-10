<!--
This delta is written against the "Incremental architecture" requirement's
target end-state after both `rescope-admission-control` (B4) and
`add-batch-priority-queueing` (B5) archive — the same layering convention
B5's own delta already established against B4's target text. Neither B4 nor
B5 is archived yet; both are merged to `develop`. This change's own archive
is expected to happen after both (`tasks.md` §6.3).

The archiver convention (a MODIFIED requirement carries the entire existing
block) is followed here against that layered target text: the "Tier and
model knobs," "Sequence-concurrency ceiling," "Knob resolution is
unavailable," and "Change is applied" scenarios are pasted unchanged from
`add-batch-priority-queueing`'s own delta. One new scenario is added for
the multi-model process-isolation guarantee this change investigates
(design.md §10).

**Option A (design.md §10) is the change-owner-approved direction** — see
`proposal.md`'s Status and `tasks.md` §2 (both decisions approved). The
scenario below was originally written at investigation time, before the
decision, deliberately mechanism-worded to avoid locking in implementation
details prematurely; one clause naming the one-process-per-model mechanism
explicitly was added once that stopped being a premature commitment and
started being the shipped implementation (`tasks.md` §3).
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

#### Scenario: Serving an additional model does not degrade any already-loaded model
- WHEN the system is already serving one model and is asked to also serve
  one or more additional models
- THEN each model is served by its own independent server process — a
  server process is never configured to serve more than one model
- AND no already-loaded model's context-length ceiling, decode
  performance, or GPU-memory budget is reduced below what that same model
  would get if it were the only model being served
- AND no model's admission-control behavior (B4's bounded wait, B5's
  priority-differentiated wait and batch-waiter cap) changes because
  another model is also being served
- AND observability signals (health, native metrics) for one model remain
  attributable to that model alone, without collision against another
  model's signals

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
