## ADDED Requirements

### Requirement: Run identity is a content hash of the run manifest
The platform SHALL identify each completed chat completion by a `run_id` that is a content hash of a
canonical serialization of the run manifest's semantics-determining fields, and SHALL treat two runs
as comparable if and only if their `run_id`s are equal. Comparability here means configuration
identity — `engine`, `model`, `runtime`, `sampling`, `request`, and each warning's `type`/`code`/
`field` — not execution-environment identity: `hardware` is bound to the manifest as provenance but
SHALL NOT participate in the `run_id` preimage, and a warning's free-text `message` SHALL NOT
participate either. The `run_id` SHALL be reproducible from the manifest and MUST NOT be a random
identifier. The platform SHALL NOT produce a cryptographic signature for the manifest in this
capability; run identity is by content address only.

#### Scenario: Identical configuration yields an identical run_id
- **WHEN** two requests produce manifests whose `engine`, `model`, `runtime`, `sampling`, `request`,
  and per-warning `type`/`code`/`field` are equal
- **THEN** both manifests have the same `run_id`

#### Scenario: A semantics field difference changes the run_id
- **WHEN** two manifests differ in any field within `engine`, `model`, `runtime`, `sampling`, or
  `request`, or in any warning's `type`, `code`, or `field`
- **THEN** their `run_id`s differ

#### Scenario: Speed, observational, and execution-environment blocks do not affect the run_id
- **WHEN** two manifests differ only in `timing`, in the `batch` block, or in the `hardware` block
- **THEN** their `run_id`s are equal

#### Scenario: A warning's free-text message does not affect the run_id
- **WHEN** two manifests' warnings are equal in `type`, `code`, and `field` for every warning, and
  differ only in a warning's `message` text
- **THEN** their `run_id`s are equal

#### Scenario: run_id is a content hash, not a signature
- **WHEN** a manifest is produced
- **THEN** its `run_id` is recomputable by hashing the same canonical preimage
- **AND** no cryptographic signature or key material is produced

### Requirement: Non-streaming chat completions carry a run identifier header
The platform SHALL emit an `X-Run-Id` HTTP header on every non-streaming
`POST /v1/chat/completions` response, carrying the run manifest's `run_id`. The platform SHALL NOT
be required to emit a manifest surface on the streaming path in this capability.

#### Scenario: Non-streaming response carries X-Run-Id
- **WHEN** a non-streaming chat completion is served
- **THEN** the HTTP response includes an `X-Run-Id` header equal to the manifest's `run_id`

#### Scenario: Streaming response carries no manifest surface in this capability
- **WHEN** a streaming chat completion is served
- **THEN** no `X-Run-Id` header and no manifest event are required of this capability

### Requirement: The run manifest attests the computation from existing signals
The platform SHALL assemble a versioned run manifest that records the provenance of the computation,
and SHALL derive it from the per-request signals the platform already produces — the resolved
request, the typed warnings, and the engine timing — without recomputing token counts, timings, or
warnings. The manifest SHALL carry a `manifest_version`. A provenance value the platform cannot
determine SHALL be omitted or recorded as null, and MUST NOT be fabricated, approximated, or
defaulted to a fictitious value.

#### Scenario: Manifest reuses existing per-request signals
- **WHEN** a chat completion has produced a resolved block, warnings, and timing
- **THEN** the manifest's corresponding fields are populated from those existing signals rather than
  recomputed

#### Scenario: Undeterminable provenance is recorded honestly
- **WHEN** a provenance value (for example a model weight hash or model revision) cannot be
  determined
- **THEN** that field is omitted or null in the manifest
- **AND** the manifest records no fabricated or approximated value in its place

#### Scenario: Manifest is versioned
- **WHEN** a manifest is produced
- **THEN** it carries a `manifest_version`

### Requirement: Batch co-batching identity is reserved but not populated
The platform SHALL define `batch.co_batched_request_ids` in the run manifest schema as the reserved
location for engine co-batch composition, and SHALL NOT populate it with engine-internal composition
data in this capability. The field MUST NOT carry fabricated composition data.

#### Scenario: The field exists in the schema
- **WHEN** the run manifest schema is defined
- **THEN** it includes `batch.co_batched_request_ids`

#### Scenario: No composition data is emitted in this capability
- **WHEN** a manifest is produced
- **THEN** `batch.co_batched_request_ids` carries no engine-derived composition data
- **AND** it carries no fabricated composition data

### Requirement: The manifest asserts comparability, not reproducibility
The platform and its documentation SHALL NOT claim end-to-end determinism or that runs are
reproducible on the basis of the seed or the manifest alone. They MAY state the manifest's
comparability property: that equal `run_id`s denote the same recorded configuration. Hardware is
recorded on the manifest as execution-environment provenance and is not part of the `run_id`
preimage; two runs of the same configuration on different hardware MAY therefore share a `run_id`,
and a consumer SHALL use the manifest's `hardware` block, not the `run_id`, to judge whether a
comparison between them is reproducible-grade or only configuration-comparable. A scoped
determinism contract is out of scope for this capability.

#### Scenario: Comparability is claimed, reproducibility is not
- **WHEN** documentation describes the run manifest
- **THEN** it states that equal `run_id`s denote the same recorded configuration
- **AND** it does not claim the server is deterministic or that runs reproduce end-to-end

#### Scenario: A recorded seed does not imply reproducibility
- **WHEN** the manifest records a `sampling.seed`
- **THEN** neither the manifest nor its documentation claims the run is reproducible from that seed
  alone

#### Scenario: Cross-hardware runs of the same configuration can share a run_id
- **WHEN** two runs share an identical `run_id`
- **THEN** their manifests MAY report different `hardware` blocks
- **AND** a consumer reads the `hardware` block, not the `run_id`, to judge whether comparing them is
  reproducible-grade or only configuration-comparable

## MODIFIED Requirements

### Requirement: Optional request seed reaches the sampler
The platform SHALL accept an optional integer `seed` on
`POST /v1/chat/completions` as a first-class field of `ChatCompletionRequest`.
When `seed` is set and the live vLLM engine builds sampling parameters, the
runtime SHALL pass that integer into the backend sampling API exactly as
received, without rewriting, clamping, or normalizing backend sentinel
semantics. When `seed` is absent or null, the runtime SHALL omit `seed` from
sampling construction so pre-OS-3 sampling behaviour is preserved for all other
parameters. Streaming and non-streaming generation SHALL use the same sampling
parameter builder. The platform MAY echo the **requested** seed on the response
as provenance — in the resolved block and in the run manifest — but MUST NOT echo
or fabricate a **backend-derived effective** seed the client did not supply, and
documentation MUST NOT claim end-to-end determinism or that runs are reproducible
from the seed alone.

#### Scenario: Seed is accepted on the wire
- **WHEN** a client sends `seed` as an integer on a chat completion request
- **THEN** the request validates successfully
- **AND** the field is not silently dropped by schema validation

#### Scenario: Seed reaches live SamplingParams unchanged
- **WHEN** a live vLLM engine builds sampling parameters for a request whose
  `seed` is set (including the value `-1`)
- **THEN** the backend sampling API receives that exact integer as `seed`
- **AND** the runtime has not mapped, rejected, or warned about the value

#### Scenario: Absent seed preserves prior sampling construction
- **WHEN** a request omits `seed` or sets it to null
- **THEN** sampling construction does not include a `seed` key
- **AND** temperature, max_tokens, top_p, and repetition_penalty rules match
  pre-OS-3 behaviour for the same other fields

#### Scenario: Stream and non-stream share the sampling builder
- **WHEN** the same request parameters including `seed` are used for streaming
  and non-streaming generation
- **THEN** both paths obtain sampling parameters from the same builder
- **AND** seed honour does not diverge by transport

#### Scenario: No response echo and no overclaim
- **WHEN** a client supplies a seed
- **THEN** the platform MAY echo that requested seed in the resolved block and in
  the run manifest as provenance
- **AND** it MUST NOT echo or fabricate a backend-derived effective seed the client
  did not supply, and no other field gains an unreviewed echo as a side effect of
  this change
- **AND** documentation does not claim the server is deterministic or that runs are
  reproducible from the seed alone

#### Scenario: Degraded engine path does not invent seed errors
- **WHEN** the vLLM engine is unavailable and streaming returns the degraded
  content-only path
- **THEN** no new seed-related warning or HTTP error is introduced
- **AND** the degraded stream remains a single content message without a
  terminal `finish_reason` of `error`

#### Scenario: Default benchmarks remain unseeded
- **WHEN** the first-party benchmark runner issues its default chat requests
- **THEN** those requests remain without a pinned `seed`
- **AND** the runner stays throughput-oriented (deterministic benchmarking is
  deferred)
