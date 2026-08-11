## 1. Probe + schema

- [x] 1.1 Add `probe_compute_capability()` (major, minor) | None in `utils/cuda_env.py` (or adjacent),
      with unit tests for supported (SM ≥ 8.0) vs unsupported classification.
- [x] 1.2 Add `deterministic: bool = False` to `ChatCompletionRequest` (and settings/startup flag).
      Keep it distinct from `strict` (DEC-052).
- [x] 1.3 Unit-test schema defaults and that `strict` + `deterministic` are independent fields.

## 2. Wiring + refusal

- [x] 2.1 When deterministic mode is on and SM ≥ 8.0: set `VLLM_BATCH_INVARIANT=1` before engine
      construction; invoke vLLM `init_batch_invariance()` as needed; run N=3 app-level warm
      generates; set manifest `runtime.batch_invariant=True`.
- [x] 2.2 Unsupported hardware + startup deterministic → fail fast at boot.
- [x] 2.3 Unsupported hardware + per-request `deterministic: true` → HTTP 400 with stable error
      code; do not execute.
- [x] 2.4 Unit tests with mocked SM probe / env for enable, boot-fail, and 400 paths (no GPU).

## 3. Conformance (C4b)

- [x] 3.1 Add gated `tests/oracle/` determinism test + `make verify-determinism` (N trials, 1 unique
      sample; same `INFERENCE_X_RUN_GPU_TESTS=1` gate as C4a).
- [x] 3.2 Brief docs note for the scoped claim (CONTRIBUTING or help text).

## 4. Validation

- [x] 4.1 `openspec validate add-deterministic-execution --strict` passes.
- [x] 4.2 `uv run pytest tests/unit -q`, `ruff`, `mypy` green without a GPU.
