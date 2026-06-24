"""Tests for CUDA environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_x.utils.cuda_env import (
    _find_venv_cuda_home,
    ensure_cuda_home,
    ensure_vllm_process_env,
    ensure_vllm_runtime_env,
)


@pytest.fixture
def _clear_cuda_env(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    path = os.environ.get("PATH", "")
    filtered = os.pathsep.join(
        p for p in path.split(os.pathsep) if "nvidia/cu" not in p
    )
    monkeypatch.setenv("PATH", filtered)


def test_find_venv_cuda_home_discovers_bundled_toolkit():
    cuda_home = _find_venv_cuda_home()
    if cuda_home is None:
        pytest.skip("nvidia/cu* toolkit not installed in this environment")
    assert (cuda_home / "bin" / "nvcc").is_file()


def test_ensure_cuda_home_sets_from_venv(monkeypatch, _clear_cuda_env):
    venv_cuda = _find_venv_cuda_home()
    if venv_cuda is None:
        pytest.skip("nvidia/cu* toolkit not installed in this environment")

    resolved = ensure_cuda_home()

    assert resolved == str(venv_cuda)
    assert os.environ["CUDA_HOME"] == str(venv_cuda)
    assert str(venv_cuda / "bin") in os.environ["PATH"].split(os.pathsep)


def test_ensure_cuda_home_respects_existing(monkeypatch, _clear_cuda_env):
    monkeypatch.setenv("CUDA_HOME", "/custom/cuda")
    assert ensure_cuda_home() == "/custom/cuda"
    assert os.environ["CUDA_HOME"] == "/custom/cuda"


def test_ensure_vllm_process_env_sets_spawn(monkeypatch, _clear_cuda_env):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    ensure_vllm_process_env()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_ensure_vllm_runtime_env_disables_flashinfer_on_wsl(monkeypatch, _clear_cuda_env):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setattr("inference_x.utils.cuda_env._is_wsl", lambda: True)
    ensure_vllm_runtime_env()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_ensure_vllm_runtime_env_respects_explicit_flashinfer(monkeypatch, _clear_cuda_env):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.setattr("inference_x.utils.cuda_env._is_wsl", lambda: True)
    ensure_vllm_runtime_env()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
