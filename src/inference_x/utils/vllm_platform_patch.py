"""Enable CUDA pinned memory on WSL when the runtime supports it.

vLLM disables pin_memory under WSL unless ``VLLM_WSL2_ENABLE_PIN_MEMORY=1``
(its supported opt-in since 0.30.0), which adds host to device copy overhead.
Recent WSL2 + CUDA driver stacks often support pinned memory, so probe at
startup and set the opt-in only when the probe succeeds. vLLM's worker
processes inherit the variable, so no import-time hook is needed.
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
    """Opt vLLM into pinned host memory on WSL when the runtime supports it."""
    global _PATCHED
    if _PATCHED or not _is_wsl():
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

    os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    _PATCHED = True
    logger.info(
        "WSL pin_memory probe passed; enabled pinned host memory for vLLM workers"
    )
