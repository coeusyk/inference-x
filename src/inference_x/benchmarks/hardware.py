"""Hardware profiler with nvidia-ml-py → nvidia-smi → CPU-only fallback chain."""
from __future__ import annotations

import os
import subprocess
from typing import Optional, TypedDict

from inference_x.benchmarks.schemas import HardwareProfile
from inference_x.utils.vllm_platform_patch import _is_wsl


class ExtendedHardwareFields(TypedDict):
    driver: Optional[str]
    cuda: Optional[str]
    cpu: Optional[str]
    wsl2: Optional[bool]


def _cpu_cores() -> int:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def _ram_total_gb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 16.0


def _profile_via_nvml() -> Optional[HardwareProfile]:
    try:
        import pynvml  # nvidia-ml-py provides this module
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem.total / (1024 ** 3)
        vram_free = mem.free / (1024 ** 3)
        pynvml.nvmlShutdown()
        return HardwareProfile(
            gpu_name=name if isinstance(name, str) else name.decode("utf-8", errors="replace"),
            vram_total_gb=round(vram_total, 2),
            vram_free_gb=round(vram_free, 2),
            cpu_cores=_cpu_cores(),
            ram_total_gb=round(_ram_total_gb(), 2),
            has_gpu=True,
        )
    except Exception:
        return None


def _profile_via_nvidia_smi() -> Optional[HardwareProfile]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        gpu_name = parts[0]
        # nvidia-smi returns MiB
        vram_total = float(parts[1]) / 1024
        vram_free = float(parts[2]) / 1024
        return HardwareProfile(
            gpu_name=gpu_name,
            vram_total_gb=round(vram_total, 2),
            vram_free_gb=round(vram_free, 2),
            cpu_cores=_cpu_cores(),
            ram_total_gb=round(_ram_total_gb(), 2),
            has_gpu=True,
        )
    except Exception:
        return None


def _cpu_only_profile() -> HardwareProfile:
    return HardwareProfile(
        gpu_name=None,
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        cpu_cores=_cpu_cores(),
        ram_total_gb=round(_ram_total_gb(), 2),
        has_gpu=False,
    )


def _driver_version() -> Optional[str]:
    """Best-effort NVIDIA driver version string, or None (never fabricated)."""
    try:
        import pynvml  # nvidia-ml-py provides this module
        pynvml.nvmlInit()
        raw = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def _cuda_version() -> Optional[str]:
    """Best-effort CUDA runtime version actually in use (torch's build), or None."""
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def _cpu_name() -> Optional[str]:
    """Best-effort CPU model name from /proc/cpuinfo (WSL2/Linux-first, DEC-platform), or None."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def extended_hardware_fields() -> ExtendedHardwareFields:
    """Best-effort driver/CUDA/CPU-name/WSL2 fields for the run manifest (Phase C, C1).

    Extends, rather than modifies, `HardwareProfile` / `profile_hardware()`:
    those are the benchmark subsystem's own established shape with its own
    consumers (advisor scoring, storage — DEC-054/055/056/057); this function
    is manifest-specific and additive. Every field is best-effort and may be
    None — never fabricated (add-run-manifest design.md D4).
    """
    return {
        "driver": _driver_version(),
        "cuda": _cuda_version(),
        "cpu": _cpu_name(),
        "wsl2": _is_wsl(),
    }


def profile_hardware() -> HardwareProfile:
    """Detect hardware using nvidia-ml-py → nvidia-smi → CPU-only fallback chain."""
    profile = _profile_via_nvml()
    if profile is not None:
        return profile

    profile = _profile_via_nvidia_smi()
    if profile is not None:
        return profile

    return _cpu_only_profile()


def _probe_torch_vram_gib() -> tuple[float | None, float | None]:
    """Return (free_gib, total_gib) from the CUDA allocator view, or (None, None)."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024**3), total / (1024**3)
    except Exception:
        pass
    return None, None


def _vram_for_utilization() -> tuple[float, float, str]:
    """VRAM basis for gpu_memory_utilization sizing (prefer torch — matches vLLM)."""
    free_gib, total_gib = _probe_torch_vram_gib()
    if free_gib is not None and total_gib is not None:
        return free_gib, total_gib, "torch"
    hw = profile_hardware()
    if hw.has_gpu:
        return hw.vram_free_gb, hw.vram_total_gb, "nvml"
    return 0.0, 0.0, "none"


def suggest_gpu_memory_utilization(
    *,
    model_count: int = 1,
    free_gib: float | None = None,
    total_gib: float | None = None,
) -> float:
    """
    Compute gpu_memory_utilization as (free_vram - buffer) / total_vram.

    For multi-model sessions, callers should pass ``model_count=1`` and re-sample
    ``free_gib`` at each engine load (see ``scale_model_config``).
    The ``model_count > 1`` branch evenly splits the budget for legacy callers only::

        ((free - buffer) / model_count) / total

    vLLM treats utilization as a fraction of total VRAM. Uses torch.cuda.mem_get_info
    when *free_gib* / *total_gib* are omitted (re-queried on every call).
    """
    buffer = float(os.getenv("INFERENCEX_VRAM_SAFETY_BUFFER_GB", "0.4"))
    if free_gib is None or total_gib is None:
        free_gib, total_gib, source = _vram_for_utilization()
        if source == "none":
            return 1.0
    usable_gb = max(0.0, free_gib - buffer)
    if model_count > 1:
        usable_gb = usable_gb / model_count
    utilization = usable_gb / total_gib
    if model_count <= 1:
        return round(max(0.50, min(0.95, utilization)), 2)
    return round(max(0.0, min(0.95, utilization)), 2)
