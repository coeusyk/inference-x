# Tasks: qualify-vllm-0-30

## 1. Pin and lock
- [x] 1.1 `pyproject.toml`: `vllm==0.30.0`
- [x] 1.2 `uv lock --upgrade-package vllm`; `uv sync --all-extras`; record the resolved diff (design D4)

## 2. Couplings
- [x] 2.1 Inventory every vLLM coupling against the 0.30.0 source (design D1)
- [x] 2.2 `utils/determinism.py`: import `init_batch_invariance` from `vllm.model_executor.determinism.batch_invariant`; update the unit test stub path (D2)
- [x] 2.3 Show live that the old `in_wsl` patch has no effect on 0.30.0 and that `VLLM_WSL2_ENABLE_PIN_MEMORY=1` works (D3)
- [x] 2.4 `utils/vllm_platform_patch.py`: set the opt-in after a passing probe; keep the probe and `INFERENCE_X_DISABLE_WSL_PIN_MEMORY`; drop `INFERENCE_X_VLLM_PATCH_APPLIED` (D3)
- [x] 2.5 Remove `scripts/install_vllm_patch.sh` and the `.pth` it installed; rewrite `tests/unit/test_vllm_platform_patch.py` for the new behavior (D3)
- [x] 2.6 Update `docs/UNDERSTANDING-INFERENCE-X.md` and the "verified against 0.22.1" comments in `engines/vllm_engine.py`

## 3. Gates
- [x] 3.1 Unit suite, ruff, mypy
- [x] 3.2 `openspec validate qualify-vllm-0-30 --strict`
- [x] 3.3 GPU oracle and determinism conformance on 0.30.0, and on 0.22.1 for the baseline
- [x] 3.4 Live smoke test

## 4. Live qualification (RTX 4060 8 GiB)
- [x] 4.1 qwen2.5-coder-1.5b: Aider acceptance 7/7
- [x] 4.2 qwen3-4b-fp8 via `local-4b`: Continue CLI acceptance 10/10 through the logging proxy
- [x] 4.3 qwen2.5-7b-awq: loads with CUDA graphs on 0.30.0; confirm 0.22.1 needed eager (D6)
- [x] 4.4 Warm-run performance baseline on both versions (D6)
- [x] 4.5 GO/NO-GO (D7): GO

## 5. Record
- [x] 5.1 DEC-066 in `docs/DECISIONS.md`
- [x] 5.2 CHANGELOG entry
