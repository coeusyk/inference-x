# Proposal: add-oracle-conformance (Phase C — C4a)

## Why

`docs/REVIEW-2026-08-03-architecture.md` §6.2b: Inference-X has hundreds of tests and **not one
asserts that the model produces correct output** — every test mocks vLLM. Colibrì's pattern is
token-exact teacher-forced validation against a `transformers` oracle, reporting exact top-1
agreement per position and *naming* floating-point near-ties rather than silently passing them.

Phase C's locked execution plan lists this as **C4a** (`add-oracle-conformance`): independent of
every other Phase C change, the first real inference test, and the gated real-GPU test tier that
later C3/C4b and D1 work reuse. Determinism conformance (`make verify-determinism`) is **C4b** and
ships with C3 — out of scope here.

## What Changes

- **`tests/oracle/` suite.** Teacher-forces `facebook/opt-125m` (already in `config/models.yaml`)
  through the Inference-X `VLLMEngine` path and raw `transformers`, asserting top-1 agreement per
  position on a fixed prompt. Floating-point near-ties are reported by position; they do not count
  as silent passes.
- **Gated real-GPU test tier.** Oracle tests require a GPU + model weights and MUST NOT run in the
  existing `checks` CI job (which mocks vLLM / runs `tests/unit` only). Introduce an explicit
  pytest marker and env opt-in so accidental `pytest tests/` on a CPU host skips cleanly rather than
  failing or downloading weights.
- **CI-safe comparison helpers.** Pure agreement / near-tie classification logic is unit-tested
  under `tests/unit/` with synthetic logits so the policy is enforced without a GPU.

## Capabilities

### Modified Capabilities

- `platform`: ADDS requirements for a gated oracle conformance suite (teacher-forced top-1 vs
  `transformers` on `opt-125m`, near-tie reporting, exclusion from the GPU-free `checks` job).

### New Capabilities

- (none)

## Impact

- **Additive and backward-compatible.** No public API, schema, or engine-construction behavior
  changes. No change to the `checks` job's GPU-free contract.
- **Affected code:** `tests/oracle/` (new), small pure helper module used by the oracle and its
  unit tests, `pyproject.toml` pytest marker registration, brief docs note (CONTRIBUTING or
  Makefile help) on how to run the gated tier.
- **Dependencies:** uses `transformers` / `torch` already pulled transitively by `vllm`; no new
  required dependency.
- **Explicitly out of scope:** C4b determinism conformance / `make verify-determinism` (ships with
  C3); Varex e2e (C5); any production API that exposes oracle results; changing numerics of the
  serving path.
