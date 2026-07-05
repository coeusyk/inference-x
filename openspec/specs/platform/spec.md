# Platform Specification

## Purpose

InferenceX is a self-hosted inference platform that exposes a stable API for LLM serving and grows incrementally from a single vLLM-backed engine into a more complete system with routing, observability, and evaluation support.
## Requirements
### Requirement: Stable API-first platform
The system SHALL expose stable HTTP interfaces before layering on optional
capabilities, and non-streaming chat completions SHALL support the same
continuous-batching concurrency as streaming completions without dropping or
corrupting any in-flight request's output.

#### Scenario: Concurrent non-streaming requests
- WHEN 2 or more non-streaming chat completion requests are in flight against the
  same loaded model at the same time
- THEN every request completes with its own, complete, non-truncated output
- AND no request's terminal output is discarded or delivered to a different
  request

#### Scenario: Engine driver failure
- WHEN the shared per-engine driver thread's call into the underlying inference
  engine raises an exception
- THEN every request currently in flight against that engine receives that
  exception instead of hanging indefinitely
- AND the engine is reported unhealthy so new requests are not accepted by a dead
  driver

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `POST /v1/chat/completions` request/response contracts are unchanged for
  both streaming and non-streaming callers
- AND all existing unit tests continue to pass unchanged

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

### Requirement: WSL2-first development
The system SHALL assume local development on WSL2 Ubuntu for Linux-only runtime components.

#### Scenario: Developer setup
- **WHEN** local setup instructions are followed
- **THEN** the runtime path targets Linux tooling inside WSL2
- **AND** Windows-native execution is not required for vLLM-bound components

### Requirement: Spec-driven changes
The project SHALL track non-trivial work through OpenSpec change artifacts.

#### Scenario: New feature is proposed
- **WHEN** a non-trivial feature or refactor is requested
- **THEN** a change folder is created under `openspec/changes/`
- **AND** the proposal, design, and task artifacts are completed before implementation proceeds

### Requirement: Model variant selection at load time
The system SHALL support grouping model configuration entries into named families
and, when an operator requests a family rather than a concrete model, SHALL select
the highest-precision variant that fits the available VRAM budget at load time.

#### Scenario: Multiple variants, one fits
- WHEN a model family has bf16, int8, and 4-bit quantized variants registered
- AND only the 4-bit variant's estimated weight size fits the available VRAM
  budget
- THEN the 4-bit variant is loaded
- AND the higher-precision variants are not loaded

#### Scenario: No variant fits
- WHEN no variant in a requested family fits the available VRAM budget
- THEN startup fails with an error naming the family and each variant's
  estimated size versus the available budget
- AND the process does not begin accepting requests

#### Scenario: Concrete model name bypasses selection
- WHEN `INFERENCE_X_LOADED_MODELS` names a concrete, registered model entry
  directly (not a family)
- THEN that exact entry is loaded, unaffected by variant selection

#### Scenario: Ungrouped model is unaffected
- WHEN a model entry has no `family` set
- THEN it loads exactly as it did before this change

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `TaskRouter` and `AdmissionController` behavior is unchanged — both
  continue to operate only on the concrete, already-resolved model name
- AND all existing unit tests continue to pass unchanged

