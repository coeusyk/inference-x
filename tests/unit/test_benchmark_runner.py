"""Unit tests for BenchmarkRunner VRAM footprint calculation."""
from __future__ import annotations

from inference_x.benchmarks.runner import _check_vram_budget, _peak_vram_footprint_gb
from inference_x.benchmarks.schemas import HardwareProfile


def _hw(
    *,
    total: float,
    free: float,
    has_gpu: bool = True,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_gb=total,
        vram_free_gb=free,
        cpu_cores=8,
        ram_total_gb=32.0,
        has_gpu=has_gpu,
    )


class TestPeakVramFootprint:
    def test_preloaded_model_nonzero_footprint(self):
        """Model already loaded: legacy delta is 0 but footprint is not."""
        before = _hw(total=6.0, free=2.78)
        after = _hw(total=6.0, free=2.78)
        assert _peak_vram_footprint_gb(before, after) == 3.22

    def test_cold_load_footprint(self):
        before = _hw(total=6.0, free=5.5)
        after = _hw(total=6.0, free=2.9)
        assert _peak_vram_footprint_gb(before, after) == 3.1

    def test_cpu_only_returns_zero(self):
        before = _hw(total=0.0, free=0.0, has_gpu=False)
        after = _hw(total=0.0, free=0.0, has_gpu=False)
        assert _peak_vram_footprint_gb(before, after) == 0.0


class TestCheckVramBudget:
    def test_no_model_path_skips_check(self):
        exceeded, warning = _check_vram_budget("m", 99.0, None, 2048, None)
        assert exceeded is False
        assert warning is None

    def test_within_budget(self):
        exceeded, warning = _check_vram_budget(
            "opt-125m", 0.01, "facebook/opt-125m", 2048, None
        )
        assert exceeded is False
        assert warning is None

    def test_exceeds_budget(self):
        exceeded, warning = _check_vram_budget(
            "opt-125m", 500.0, "facebook/opt-125m", 2048, None
        )
        assert exceeded is True
        assert warning is not None
        assert "opt-125m" in warning
