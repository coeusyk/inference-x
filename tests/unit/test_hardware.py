"""Unit tests for hardware profiler with mocked pynvml / nvidia-smi."""
from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from inference_x.benchmarks.hardware import (
    _cpu_only_profile,
    _profile_via_nvidia_smi,
    _profile_via_pynvml,
    profile_hardware,
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


# ---------------------------------------------------------------------------
# pynvml path tests
# ---------------------------------------------------------------------------

class TestProfileViaPynvml:
    def test_6gb_gpu(self):
        mock_pynvml = _make_pynvml_mock("NVIDIA GeForce RTX 3060", 6144, 5120)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_pynvml()
        assert profile is not None
        assert profile.has_gpu is True
        assert profile.gpu_name == "NVIDIA GeForce RTX 3060"
        assert abs(profile.vram_total_gb - 6.0) < 0.1
        assert abs(profile.vram_free_gb - 5.0) < 0.1

    def test_24gb_gpu(self):
        mock_pynvml = _make_pynvml_mock("NVIDIA RTX 3090", 24576, 20480)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_pynvml()
        assert profile is not None
        assert profile.has_gpu is True
        assert abs(profile.vram_total_gb - 24.0) < 0.1
        assert abs(profile.vram_free_gb - 20.0) < 0.1

    def test_pynvml_import_error_returns_none(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            profile = _profile_via_pynvml()
        assert profile is None

    def test_pynvml_runtime_error_returns_none(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("nvml init failed")
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = _profile_via_pynvml()
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
    def test_uses_pynvml_when_available(self):
        mock_pynvml = _make_pynvml_mock("Test GPU", 8192, 7168)
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            profile = profile_hardware()
        assert profile.has_gpu is True
        assert profile.gpu_name == "Test GPU"

    def test_falls_back_to_nvidia_smi_when_pynvml_absent(self):
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
