"""Enable CUDA pinned memory on WSL when the runtime supports it.

vLLM disables pin_memory whenever ``in_wsl()`` is true, which adds host→device
copy overhead. Recent WSL2 + CUDA driver stacks often support pinned memory;
probe at startup and patch vLLM only when the probe succeeds.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_PATCHED = False


def _is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _probe_cuda_pin_memory() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        host = torch.empty(4096, device="cpu", pin_memory=True)
        host.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def apply() -> None:
    """Patch vLLM WSL detection so pin_memory stays enabled when supported."""
    global _PATCHED
    if _PATCHED:
        return

    if os.environ.get("INFERENCE_X_VLLM_PATCH_APPLIED") == "1":
        _patch_vllm_wsl_flag()
        _PATCHED = True
        return

    if not _is_wsl():
        return

    if os.environ.get("INFERENCE_X_DISABLE_WSL_PIN_MEMORY", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("WSL pin_memory override disabled via INFERENCE_X_DISABLE_WSL_PIN_MEMORY")
        return

    if not _probe_cuda_pin_memory():
        logger.info(
            "WSL pin_memory probe failed; keeping vLLM default (pin_memory=False)"
        )
        return

    os.environ["INFERENCE_X_VLLM_PATCH_APPLIED"] = "1"
    _patch_vllm_wsl_flag()
    _PATCHED = True
    logger.info(
        "WSL pin_memory probe passed; enabled pinned host memory for vLLM workers"
    )


def _patch_vllm_wsl_flag() -> None:
    import vllm.platforms.interface as vllm_platform

    vllm_platform.in_wsl = lambda: False  # type: ignore[method-assign]
