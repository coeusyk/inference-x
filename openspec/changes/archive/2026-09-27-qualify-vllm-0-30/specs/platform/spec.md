## ADDED Requirements

### Requirement: vLLM runtime pinned to an exact qualified release
The platform SHALL pin the vLLM dependency to one exact release in `pyproject.toml`, and that release SHALL be one that passed a recorded qualification on the project's reference hardware. A qualification SHALL record the old and candidate versions, the resolved dependency changes, the unit, oracle and determinism suites, the Aider and Continue acceptance results, and a warm-run performance baseline per tested model class. Moving to a different vLLM release SHALL be its own change and SHALL NOT be combined with unrelated feature work.

#### Scenario: Dependency declaration
- **WHEN** the project's dependencies are read
- **THEN** vLLM is declared with an exact `==` version, not a lower bound
- **AND** `uv.lock` resolves that same version

#### Scenario: Version reaches provenance
- **WHEN** a non-streaming request is served with `include_manifest: true`
- **THEN** the manifest's `engine.backend_version` reports the installed vLLM version, read from package metadata

### Requirement: WSL2 pinned memory enabled only through vLLM's supported opt-in after a successful probe
On WSL2, the platform SHALL enable pinned host memory for vLLM only by setting vLLM's own opt-in environment variable (`VLLM_WSL2_ENABLE_PIN_MEMORY=1`) before any engine is constructed, and only after a startup probe successfully copies a pinned host tensor to the GPU. The platform SHALL NOT patch vLLM internals or install import hooks into the environment to change vLLM's platform detection. Setting `INFERENCE_X_DISABLE_WSL_PIN_MEMORY` to a true value SHALL leave vLLM's default in place.

#### Scenario: Probe passes on WSL2
- **WHEN** the process starts on WSL2 and the pinned-memory probe succeeds
- **THEN** `VLLM_WSL2_ENABLE_PIN_MEMORY` is `1` in the process environment before engine construction
- **AND** vLLM worker processes inherit it and report pinned memory as available

#### Scenario: Probe fails
- **WHEN** the process starts on WSL2 and the pinned-memory probe fails
- **THEN** the opt-in is not set and vLLM keeps its default of no pinned memory

#### Scenario: Operator opts out
- **WHEN** `INFERENCE_X_DISABLE_WSL_PIN_MEMORY` is set to a true value
- **THEN** the opt-in is not set, whatever the probe would return

#### Scenario: Not WSL
- **WHEN** the process does not run under WSL
- **THEN** the platform does not set the opt-in and vLLM's own platform logic applies
