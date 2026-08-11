## ADDED Requirements

### Requirement: GET /v1/plan reports per-model sizing without loading an engine
The platform SHALL expose `GET /v1/plan`, which reports, for every model registered in the model
registry, the estimated weight footprint, estimated KV-cache footprint at the model's configured
`max_model_len`, the resolved `gpu_memory_utilization`, and the resolved VRAM-tier knobs
(`block_size`, `kv_cache_dtype`, `enable_prefix_caching`, `max_num_seqs`,
`max_num_batched_tokens`) the platform would use if that model were loaded on the current machine.
The platform SHALL compute these values using the same sizing functions the engine construction
path uses, and SHALL NOT start, stop, or reconfigure any engine to answer this request.

#### Scenario: Plan reports every registered model
- **WHEN** a client sends `GET /v1/plan`
- **THEN** the response includes one entry per model in the model registry
- **AND** no engine is started, stopped, or reconfigured as a result of the request

#### Scenario: Plan sizing matches the engine's own sizing computation
- **WHEN** a model's `plan` entry reports `gpu_memory_utilization`
- **THEN** that value is produced by the same sizing function the engine construction path calls
  for that model, not a separately maintained approximation

### Requirement: GET /v1/doctor reports read-only environment and per-model fit readiness
The platform SHALL expose `GET /v1/doctor`, which reports the probed GPU memory (total and free),
the hardware profile, and, for every registered model, whether that model fits on the currently
probed VRAM. When a model does not fit, the platform SHALL report the specific reason. The platform
SHALL NOT start an engine to produce this report.

#### Scenario: Doctor reports environment and per-model fit
- **WHEN** a client sends `GET /v1/doctor`
- **THEN** the response includes the probed GPU memory and hardware profile
- **AND** the response includes a fit result for every registered model

#### Scenario: A model that does not fit reports why
- **WHEN** a registered model's estimated footprint exceeds the probed free VRAM
- **THEN** that model's `doctor` entry reports it does not fit
- **AND** the entry includes the specific reason the platform's existing fit check produced

#### Scenario: A model that fits reports success without a failure reason
- **WHEN** a registered model's estimated footprint fits within the probed VRAM
- **THEN** that model's `doctor` entry reports it fits
- **AND** the entry carries no failure reason

#### Scenario: Doctor does not start an engine
- **WHEN** a client sends `GET /v1/doctor`
- **THEN** no engine is started, stopped, or reconfigured as a result of the request
