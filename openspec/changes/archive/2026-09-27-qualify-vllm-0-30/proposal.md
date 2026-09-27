# Proposal: qualify-vllm-0-30

## Why

InferenceX has run on vLLM 0.22.1 since the AsyncLLM migration, but `pyproject.toml` only said `vllm>=0.6.0`, so the version that actually ran was whatever `uv.lock` happened to hold. vLLM 0.30.0 is the current release. Between the two, upstream moved to PyTorch 2.13 (0.27) and Transformers 5.15 (0.28), made Model Runner V2 the default (0.29), and changed how WSL2 pinned memory is enabled. V1-1 (the llama.cpp backend) is next, and it should start from a vLLM baseline that has been checked on this hardware, not one that drifted.

The owner asked for this upgrade as its own compatibility change: pin exactly 0.30.0 (not main, and not waiting for 0.31), qualify it against the acceptance clients and the correctness suites, and adopt it only if it passes. If qualification had failed materially, InferenceX would have stayed on 0.22.1 with the reason recorded. This change is deliberately not combined with V1-1.

## What changes

- **Exact pin.** `pyproject.toml` pins `vllm==0.30.0`. `uv.lock` was regenerated with `uv lock --upgrade-package vllm`, so only vLLM and what it requires moved. The resolved dependency diff is recorded in `design.md`.
- **Batch-invariant import path.** vLLM moved `init_batch_invariance` from `vllm.model_executor.layers.batch_invariant` to `vllm.model_executor.determinism.batch_invariant`. `utils/determinism.py` imports from the new path. Without this, starting a process in deterministic mode would fail with an import error.
- **WSL2 pinned memory uses vLLM's supported opt-in.** vLLM 0.30.0 disables pinned host memory under WSL2 unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. The old InferenceX patch replaced `vllm.platforms.interface.in_wsl` with a function returning False. On 0.30.0 that has no effect, because `platforms/cuda.py` imports `in_wsl` by name and checks the kernel version and the env var itself. `utils/vllm_platform_patch.py` now sets `VLLM_WSL2_ENABLE_PIN_MEMORY=1` when WSL is detected and the existing pinned-memory probe passes. The probe and the `INFERENCE_X_DISABLE_WSL_PIN_MEMORY` opt-out are unchanged. vLLM's worker processes inherit the env var, so `scripts/install_vllm_patch.sh` (which installed a `.pth` hook into site-packages to repeat the old patch in every worker) is removed, along with its `INFERENCE_X_VLLM_PATCH_APPLIED` marker.
- **Docs and comments** that described the old patch or said a coupling was "verified against 0.22.1" now describe the current mechanism or say they were re-checked on 0.30.0. Historical records (archived changes, earlier DEC entries) are left as they were.

No API, schema, config, or response-shape change. `backend_version` in the run manifest reports `0.30.0`, which changes `run_id` for otherwise identical requests. That is the intended behavior: the backend version is part of run identity (add-run-manifest, the change that introduced the run manifest).

## Result

**GO.** Correctness, Aider, Continue tool calling, determinism and the oracle, metrics and timing truthfulness, WSL2 startup, and all three tested model classes (dense bf16, FP8, AWQ) held on vLLM 0.30.0 on the RTX 4060 8 GiB desktop. The full matrix, the performance baseline, and the two measured regressions (lower KV capacity at the same `gpu_memory_utilization`, slower FP8 prefill) are in `design.md`. Neither regression breaks a configured model or an acceptance client. 0.30.0 is the new baseline, recorded in `docs/DECISIONS.md` as DEC-066 (vLLM 0.30.0 as the qualified, exactly pinned baseline).

## Out of scope

- V1-1 (llama.cpp backend). It follows as a separate change.
- Admission restoration. It stays deferred per `docs/INFERENCEX-EVOLUTION.md` section 10.3 and resumes only on one of the triggers documented there.
- Changing any model's `gpu_memory_utilization` or context to win back KV capacity. That is recorded as a follow-up.
