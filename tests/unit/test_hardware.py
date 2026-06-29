"""Unit tests for hardware profiler with mocked nvidia-ml-py / nvidia-smi."""
from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from inference_x.benchmarks.hardware import (
    _cpu_only_profile,
    _profile_via_nvidia_smi,
    _profile_via_nvml,
    profile_hardware,
    suggest_gpu_memory_utilization,
)
from inference_x.benchmarks.schemas import HardwareProfile


# ---------------------------------------------------------------------------
# Helpers for building pynvml mock
# ---------------------------------------------------------------------------

def _make_pynvml_mock(gpu_name: str, total_mib: int, free_mib: int) -> ModuleType:
    pynvml = MagicMock()
    handle = MagicMock()
    mem_info = MagicMock()
    mem_info.total = total_mib * 1024 * 1024
    mem_info.free = free_mib * 1024 * 1024
    pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    pynvml.nvmlDeviceGetName.return_value = gpu_name
    pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    return pynvml


def _hw_profile(
    *,
    total: float,
    free: float,
    has_gpu: bool = True,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU" if has_gpu else None,
        vram_total_gb=total,
        vram_free_gb=free,
        cpu_cores=8,
        ram_total_gb=16.0,
        has_gpu=has_gpu,
    )


@pytest.fixture()
def mock_hw(monkeypatch):
    """Patch VRAM probes for suggest_gpu_memory_utilization tests."""

    def _apply(
        *,
        total: float = 8.0,
        free: float = 6.93,
        has_gpu: bool = True,
    ) -> HardwareProfile:
        profile = _hw_profile(total=total, free=free, has_gpu=has_gpu)
        monkeypatch.setattr(
            "inference_x.benchmarks.hardware.profile_hardware",
            lambda: profile,
        )
        if has_gpu:
            monkeypatch.setattr(
                "inference_x.benchmarks.hardware._probe_torch_vram_gib",
                lambda: (free, total),
            )
        else:
            monkeypatch.setattr(
                "inference_x.benchmarks.hardware._probe_torch_vram_gib",
                lambda: (None, None),
            )
        return profile

    return _apply


# ---------------------------------------------------------------------------
# pynvml path tests
# ---------------------------------------------------------------------------

class TestProfileViaNvml:
    def test_6gb_gpu(self):
        mock_pynvml = _make_pynvml_mock("NVIDIA GeForce RTX 3060", 6144, 5120)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_nvml()
        assert profile is not None
        assert profile.has_gpu is True
        assert profile.gpu_name == "NVIDIA GeForce RTX 3060"
        assert abs(profile.vram_total_gb - 6.0) < 0.1
        assert abs(profile.vram_free_gb - 5.0) < 0.1

    def test_24gb_gpu(self):
        mock_pynvml = _make_pynvml_mock("NVIDIA RTX 3090", 24576, 20480)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_nvml()
        assert profile is not None
        assert profile.has_gpu is True
        assert abs(profile.vram_total_gb - 24.0) < 0.1
        assert abs(profile.vram_free_gb - 20.0) < 0.1

    def test_nvml_import_error_returns_none(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profile = _profile_via_nvml()
        assert profile is None

    def test_nvml_runtime_error_returns_none(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("nvml init failed")
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_nvml()
        assert profile is None


# ---------------------------------------------------------------------------
# nvidia-smi path tests
# ---------------------------------------------------------------------------

class TestProfileViaNvidiaSmi:
    def test_parses_valid_output(self):
        mock_output = "NVIDIA GeForce RTX 3060, 6144, 5120\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output
        with patch("subprocess.run", return_value=mock_result):
            profile = _profile_via_nvidia_smi()
        assert profile is not None
        assert profile.has_gpu is True
        assert profile.gpu_name == "NVIDIA GeForce RTX 3060"
        assert abs(profile.vram_total_gb - 6.0) < 0.1
        assert abs(profile.vram_free_gb - 5.0) < 0.1

    def test_nonzero_returncode_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            profile = _profile_via_nvidia_smi()
        assert profile is None

    def test_subprocess_exception_returns_none(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            profile = _profile_via_nvidia_smi()
        assert profile is None


# ---------------------------------------------------------------------------
# CPU-only fallback
# ---------------------------------------------------------------------------

class TestCpuOnlyProfile:
    def test_returns_no_gpu_profile(self):
        profile = _cpu_only_profile()
        assert profile.has_gpu is False
        assert profile.gpu_name is None
        assert profile.vram_total_gb == 0.0
        assert profile.vram_free_gb == 0.0
        assert profile.cpu_cores >= 1
        assert profile.ram_total_gb >= 0.0

    def test_cpu_cores_populated(self):
        profile = _cpu_only_profile()
        assert isinstance(profile.cpu_cores, int)
        assert profile.cpu_cores >= 1


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class TestProfileHardwareFallbackChain:
    def test_uses_nvml_when_available(self):
        mock_pynvml = _make_pynvml_mock("Test GPU", 8192, 7168)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = profile_hardware()
        assert profile.has_gpu is True
        assert profile.gpu_name == "Test GPU"

    def test_falls_back_to_nvidia_smi_when_nvml_absent(self):
        mock_output = "Test GPU, 8192, 7168\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", return_value=mock_result):
                profile = profile_hardware()
        assert profile.has_gpu is True

    def test_falls_back_to_cpu_only_when_both_fail(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                profile = profile_hardware()
        assert profile.has_gpu is False
        assert profile.vram_total_gb == 0.0

    def test_profile_is_hardware_profile_instance(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                profile = profile_hardware()
        assert isinstance(profile, HardwareProfile)


# ---------------------------------------------------------------------------
# GPU memory utilization suggestion
# ---------------------------------------------------------------------------

class TestSuggestGpuMemoryUtilization:
    def test_normal_case(self, mock_hw):
        mock_hw(total=8.0, free=6.93)
        assert suggest_gpu_memory_utilization() == 0.82

    def test_clamp_low(self, mock_hw):
        mock_hw(total=8.0, free=1.0)
        assert suggest_gpu_memory_utilization() == 0.50

    def test_clamp_high(self, mock_hw):
        # (8.0 - 0.4) / 8.0 = 0.95
        mock_hw(total=8.0, free=8.0)
        assert suggest_gpu_memory_utilization() == 0.95

    def test_cpu_only(self, mock_hw):
        mock_hw(total=0.0, free=0.0, has_gpu=False)
        assert suggest_gpu_memory_utilization() == 1.0

    def test_env_var_buffer(self, mock_hw, monkeypatch):
        monkeypatch.setenv("INFERENCEX_VRAM_SAFETY_BUFFER_GB", "0.8")
        mock_hw(total=8.0, free=6.93)
        assert suggest_gpu_memory_utilization() == 0.77

    def test_auto_splits_budget_across_model_count(self, mock_hw):
        mock_hw(total=8.0, free=6.93)
        expected = ((6.93 - 0.4) / 2) / 8.0
        assert suggest_gpu_memory_utilization(
            model_count=2, free_gib=6.93, total_gib=8.0
        ) == pytest.approx(expected, abs=0.01)
