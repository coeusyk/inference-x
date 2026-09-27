## ADDED Requirements

### Requirement: llama.cpp served through an external llama-server process
The platform SHALL support `engine: llama_cpp` model entries that serve a GGUF file through an external `llama-server` process started and owned by the engine, behind the same `/v1/chat/completions` endpoint and request contract as vLLM entries. The platform SHALL NOT load llama.cpp in-process. One InferenceX process SHALL serve one model, and a llama.cpp entry SHALL run one sequence at a time. The `llama-server` process SHALL listen only on the loopback interface and SHALL be stopped when the engine shuts down or the InferenceX process dies.

#### Scenario: GGUF model serves a chat completion
- **WHEN** InferenceX is started with a `llama_cpp` entry whose GGUF file and `llama-server` binary are available
- **THEN** non-streaming and streaming chat completions succeed through `/v1/chat/completions`
- **AND** `usage` reports the token counts `llama-server` measured

#### Scenario: Startup cannot complete
- **WHEN** the binary is missing, the GGUF file cannot be resolved, or `llama-server` exits or never becomes healthy during startup
- **THEN** InferenceX fails startup before serving and the error names the cause

#### Scenario: llama-server dies after startup
- **WHEN** the `llama-server` process exits while InferenceX is running
- **THEN** chat requests fail with HTTP 503 and code `engine_unavailable`
- **AND** `/health` returns 503

### Requirement: Engines are constructed through one factory
The platform SHALL construct engines only through `engines/registry.create_engine`, which dispatches on the model entry's `engine`. Backend-specific sizing (vLLM tier knobs and VRAM fit checks) SHALL run only in that backend's branch.

#### Scenario: Composition root
- **WHEN** the application builds its engine
- **THEN** it calls the factory and does not construct a concrete engine class itself

### Requirement: llama.cpp entries are validated against what the backend supports
A `llama_cpp` entry SHALL name a GGUF file (a Hugging Face repo id with `gguf_file`, or a local `.gguf` path), SHALL set `max_model_len`, and SHALL use `max_num_seqs: 1`. Setting a vLLM-only field or `tool_call_parser` on a `llama_cpp` entry, or `gguf_file` or `n_gpu_layers` on a `vllm` entry, SHALL be a configuration error rather than being ignored.

#### Scenario: vLLM-only field on a llama.cpp entry
- **WHEN** a `llama_cpp` entry sets `gpu_memory_utilization`
- **THEN** loading the model configuration fails with a validation error naming the field

### Requirement: Unsupported llama.cpp capabilities are explicit
For llama.cpp entries the platform SHALL reject tool requests with 400 `tool_calling_unsupported`, SHALL refuse to start in deterministic mode, SHALL report `timing` as null because `llama-server` does not measure queue time, and SHALL send llama.cpp-only samplers that InferenceX does not expose as disabled, so the sampling that runs is the sampling `resolved` reports.

#### Scenario: Tool request to a GGUF model
- **WHEN** a request with `tools` is routed to a `llama_cpp` entry
- **THEN** the platform responds 400 with code `tool_calling_unsupported` before admission

#### Scenario: Deterministic startup with a GGUF model
- **WHEN** InferenceX is started with deterministic mode enabled and a `llama_cpp` model
- **THEN** startup fails instead of serving without the guarantee

### Requirement: Manifest records the actual backend and GGUF identity
For llama.cpp runs the run manifest SHALL report `engine.backend` as `llama.cpp` and `engine.backend_version` as the running server's build id, and SHALL identify the model by its GGUF file: repository and snapshot revision when resolved from the Hub, file name, the sha256 of the loaded file, and the quantization the server reports. Fields that describe vLLM-only runtime state, and `hardware.cuda`, SHALL be null for llama.cpp runs. Adding llama.cpp SHALL NOT change the `run_id` of any vLLM request.

#### Scenario: Provenance of a GGUF run
- **WHEN** a non-streaming request to a `llama_cpp` entry sets `include_manifest: true`
- **THEN** `manifest.model.weights_sha256` equals the sha256 of the GGUF file `llama-server` loaded
- **AND** `manifest.engine.backend` is `llama.cpp`

#### Scenario: vLLM identity unchanged
- **WHEN** a vLLM request that produced a given `run_id` before this change is repeated with the same configuration
- **THEN** it produces the same `run_id`

### Requirement: Planning and diagnostics are per backend
`/v1/plan` and `/v1/doctor` SHALL report each model's `backend`. For llama.cpp entries they SHALL use llama.cpp's own fitting tool rather than vLLM's estimator, SHALL report null for vLLM-only quantities, and SHALL report a missing binary or GGUF file instead of estimating around it. Neither endpoint SHALL download a model.

#### Scenario: GGUF model that only partially fits
- **WHEN** llama.cpp's fitting tool cannot place every layer on the GPU at the configured context
- **THEN** `/v1/doctor` reports `fits: false` with the number of GPU layers in `reason`
