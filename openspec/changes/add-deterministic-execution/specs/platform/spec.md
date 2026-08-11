## ADDED Requirements

### Requirement: Deterministic mode wires batch-invariant execution on supported GPUs
The platform SHALL accept a first-class `deterministic` boolean on chat-completion requests
(default false) and a process startup setting that enables deterministic mode for the process.
When deterministic mode is active and the GPU compute capability is SM ≥ 8.0, the platform SHALL
enable vLLM batch-invariant mode (`VLLM_BATCH_INVARIANT`) before serving, run a small
application-level warmup, and record `runtime.batch_invariant: true` on the run manifest.
`deterministic` SHALL NOT be an alias of `strict` (DEC-052).

#### Scenario: Deterministic request on supported hardware enables batch-invariant mode
- **WHEN** a client sends `deterministic: true` on a host with SM ≥ 8.0
- **THEN** the platform enables vLLM batch-invariant mode for that serving path
- **AND** the run manifest's `runtime.batch_invariant` is true
- **AND** `strict` is unchanged in meaning

#### Scenario: Startup deterministic setting fails fast on unsupported hardware
- **WHEN** the process is started with deterministic mode enabled on a host with SM &lt; 8.0 or no
  CUDA GPU
- **THEN** startup fails before serving traffic
- **AND** the process does not silently run without batch-invariant mode

#### Scenario: Per-request deterministic flag refuses on unsupported hardware
- **WHEN** a client sends `deterministic: true` on a host with SM &lt; 8.0 or no CUDA GPU
- **THEN** the platform responds with HTTP 400
- **AND** it does not execute the request under a non-deterministic configuration

### Requirement: Gated determinism conformance verifies one unique sample over N trials
The platform SHALL maintain a gated GPU conformance check (`make verify-determinism`) that runs N
identical seeded trials under `deterministic: true` and passes only when exactly one unique sample
is observed. The check SHALL use the same GPU opt-in gate as the oracle suite and SHALL NOT run in
the GPU-free `checks` job.

#### Scenario: Verify-determinism passes with one unique sample
- **WHEN** the gated determinism suite runs successfully on supported hardware
- **THEN** N seeded deterministic trials produce exactly one unique completion sample

#### Scenario: Verify-determinism is excluded from checks
- **WHEN** the automated `checks` verification runs
- **THEN** it does not execute the determinism conformance suite

### Requirement: Determinism documentation states only a scoped claim
Documentation for deterministic mode SHALL state that the guarantee is scoped to repeated
identical trials on the same hardware and vLLM version under batch-invariant mode, and SHALL NOT
claim end-to-end or cross-machine reproducibility.

#### Scenario: Docs avoid end-to-end reproducibility claims
- **WHEN** documentation describes `deterministic: true`
- **THEN** it states the scoped same-hardware / same-vLLM-version claim
- **AND** it does not claim end-to-end reproducibility
