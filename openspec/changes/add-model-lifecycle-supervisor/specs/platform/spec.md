## ADDED Requirements

### Requirement: One endpoint serves every registered model through per-model workers
The platform SHALL provide a supervisor that serves the OpenAI-compatible endpoint for all registered models and routes each request to a worker process serving only the request's model, resolved by registered name or alias. Each worker SHALL be an unchanged single-model InferenceX process, so admission, `resolved`, `warnings`, manifests and `run_id` are produced by the worker exactly as when it runs alone. The supervisor SHALL forward request bodies unchanged and SHALL pass the worker's status, body and client-relevant headers back unchanged. The supervisor SHALL NOT load a model or hold GPU memory itself.

#### Scenario: Request for a model that is not loaded
- **WHEN** a chat completion names a registered model with no running worker
- **THEN** the supervisor starts that model's worker, waits until it is healthy, and forwards the request
- **AND** concurrent requests for the same model wait for that one load instead of starting another

#### Scenario: Unregistered model
- **WHEN** a request names a model that is neither a registered name nor an alias
- **THEN** the supervisor responds 404 with code `model_not_found` and does not route it to any other model

### Requirement: Loaded models are bounded and never stopped mid-request
The supervisor SHALL run at most `INFERENCE_X_MAX_LOADED_MODELS` workers (default 1). To load another model it SHALL stop the least recently used worker that has no requests in flight, counting a request in flight until its response, streaming included, has been fully sent. If no worker becomes idle within `INFERENCE_X_SWITCH_WAIT_S`, the request SHALL fail with 503 `model_busy` and a `Retry-After` header. Models SHALL stay loaded until evicted or explicitly unloaded.

#### Scenario: Switching models while a request is running
- **WHEN** a request for model B arrives while the only loaded worker, model A, is serving a request
- **THEN** A's request completes normally before A's worker is stopped
- **AND** B's request is then served by a new worker

### Requirement: A worker failure is isolated and reported
A worker that fails to start SHALL make the triggering request fail with 503 `model_load_failed`, including the exit code and the tail of the worker's log. A worker that exits while running SHALL be recorded as failed with its exit code, requests being forwarded to it SHALL fail with 503 `engine_unavailable`, other workers SHALL keep serving, and the next request for that model SHALL start a new worker. The supervisor SHALL NOT restart workers on its own.

#### Scenario: Worker killed while another model is loaded
- **WHEN** one of two loaded workers is killed
- **THEN** requests for the other model keep succeeding
- **AND** the lifecycle listing reports the killed model as failed with its exit code

### Requirement: Lifecycle is observable and controllable
The supervisor SHALL expose `GET /v1/lifecycle` listing every registered model with its state (`unloaded`, `loading`, `loaded`, `failed`) and, for a loaded model, its worker's pid, port, start time, last use and requests in flight. `POST /v1/lifecycle/load` and `POST /v1/lifecycle/unload` SHALL load and unload a named model under the same rules as request-driven loading; unloading a model whose requests do not finish within the switch wait SHALL fail with 409 `model_busy`. Workers SHALL be stopped when the supervisor stops, including when the supervisor process dies.

#### Scenario: Supervisor killed
- **WHEN** the supervisor process is killed
- **THEN** its worker processes, and any `llama-server` they own, exit as well
