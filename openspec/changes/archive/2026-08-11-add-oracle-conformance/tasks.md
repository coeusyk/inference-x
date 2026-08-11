## 1. Comparison helpers (CI-safe)

- [x] 1.1 Add a pure near-tie / agreement classifier (exact / near_tie / mismatch) over
      per-position top-1/top-2 token ids and logit margins, with a documented epsilon default.
- [x] 1.2 Unit-test the classifier under `tests/unit/`: exact agreement, named near-tie on
      disagreement-with-small-margin, hard mismatch failure signal, and "near-tie is not counted as
      exact."

## 2. Gated oracle suite

- [x] 2.1 Register a `gpu` pytest marker in `pyproject.toml` and document
      `INFERENCE_X_RUN_GPU_TESTS=1` as the opt-in for real-GPU suites.
- [x] 2.2 Add `tests/oracle/` with a skip guard (marker + env + CUDA availability) that skips
      cleanly when the gate is closed.
- [x] 2.3 Implement teacher-forced comparison for `opt-125m`: Inference-X `VLLMEngine` prompt
      logprobs vs `transformers` forward logits on a fixed plaintext prompt; print exact count and
      near-tie list; fail on hard mismatches only.
- [x] 2.4 Add `make oracle` (or equivalent help text) documenting how to run the gated suite.

## 3. Validation

- [x] 3.1 `openspec validate add-oracle-conformance --strict` passes.
- [x] 3.2 `uv run pytest tests/unit -q`, `ruff`, and `mypy` stay green without a GPU; oracle tests
      are not collected by `tests/unit`.
