"""Hardware profiler with nvidia-ml-py → nvidia-smi → CPU-only fallback chain."""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from inference_x.benchmarks.schemas import HardwareProfile


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


def profile_hardware() -> HardwareProfile:
    """Detect hardware using nvidia-ml-py → nvidia-smi → CPU-only fallback chain."""
    profile = _profile_via_nvml()
    if profile is not None:
        return profile

    profile = _profile_via_nvidia_smi()
    if profile is not None:
        return profile

    return _cpu_only_profile()
