"""Shared pytest fixtures for unit tests."""
from __future__ import annotations

import pytest

from inference_x.benchmarks.schemas import HardwareProfile


@pytest.fixture
def make_hardware():
    def _make(
        gpu_name: str = "RTX 3060 Laptop GPU",
        vram_total_gb: float = 6.0,
        vram_free_gb: float = 4.9,
        *,
        has_gpu: bool = True,
    ) -> HardwareProfile:
        return HardwareProfile(
            gpu_name=gpu_name,
            vram_total_gb=vram_total_gb,
            vram_free_gb=vram_free_gb,
            cpu_cores=8,
            ram_total_gb=32.0,
            has_gpu=has_gpu,
        )

    return _make
