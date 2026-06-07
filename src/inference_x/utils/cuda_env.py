"""CUDA and vLLM runtime environment helpers for WSL2 / venv-only toolkit installs."""

from __future__ import annotations

import logging
import os
import shutil
import site
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _find_venv_cuda_home() -> Path | None:
    """Return CUDA toolkit root bundled under site-packages/nvidia/cu*."""
    search_roots: list[Path] = []
    for entry in site.getsitepackages():
        search_roots.append(Path(entry))
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(Path(user_site))

    candidates: list[Path] = []
    for root in search_roots:
        nvidia_dir = root / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for cu_dir in nvidia_dir.glob("cu*"):
            nvcc = cu_dir / "bin" / "nvcc"
            if nvcc.is_file():
                candidates.append(cu_dir)

    if not candidates:
        return None

    # Prefer the newest bundled toolkit when multiple are present.
    return sorted(candidates, key=lambda p: p.name)[-1]


def ensure_cuda_home() -> str | None:
    """Ensure CUDA_HOME (and nvcc on PATH) for FlashInfer JIT in vLLM.

    vLLM's FlashInfer sampler JIT-compiles CUDA kernels and requires nvcc.
    On WSL2, the full CUDA toolkit is often absent from /usr/local/cuda, but
    vllm pulls nvidia-cuda-nvcc wheels into site-packages/nvidia/cu*/.

    Returns the resolved CUDA_HOME, or None if already configured externally.
    """
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if shutil.which("nvcc"):
        nvcc_path = Path(shutil.which("nvcc") or "")
        cuda_home = nvcc_path.parent.parent
        os.environ["CUDA_HOME"] = str(cuda_home)
        logger.info("CUDA_HOME set from nvcc on PATH: %s", cuda_home)
        return str(cuda_home)

    venv_cuda = _find_venv_cuda_home()
    if venv_cuda is None:
        return None

    os.environ["CUDA_HOME"] = str(venv_cuda)
    bin_dir = str(venv_cuda / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path

    logger.info(
        "CUDA_HOME set from venv CUDA toolkit (FlashInfer JIT): %s", venv_cuda
    )
    return str(venv_cuda)


from inference_x.utils.vllm_platform_patch import apply as apply_vllm_platform_patch


def ensure_vllm_runtime_env(*, pool_size: int = 1) -> str | None:
    """Apply WSL2-friendly defaults before vLLM engine initialization."""
    apply_vllm_platform_patch()
    cuda_home = ensure_cuda_home()

    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return cuda_home

    if _is_wsl():
        # FlashInfer JIT needs a full, matching CUDA toolkit. WSL2 often has GPU
        # drivers but no system CUDA install; bundled nvcc headers may not match
        # FlashInfer's CCCL includes. PyTorch-native sampling avoids JIT entirely.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        logger.info(
            "WSL detected: disabled FlashInfer sampler (VLLM_USE_FLASHINFER_SAMPLER=0)"
        )

    return cuda_home
