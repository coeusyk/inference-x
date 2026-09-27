"""Tests for WSL pin_memory opt-in."""

from __future__ import annotations

import inference_x.utils.vllm_platform_patch as patch


def _wsl(monkeypatch, probe: bool) -> None:
    monkeypatch.setattr(patch, "_PATCHED", False)
    monkeypatch.setattr(patch, "_is_wsl", lambda: True)
    monkeypatch.setattr(patch, "_probe_cuda_pin_memory", lambda: probe)
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.delenv("INFERENCE_X_DISABLE_WSL_PIN_MEMORY", raising=False)


def test_apply_sets_vllm_opt_in_when_probe_passes(monkeypatch):
    _wsl(monkeypatch, probe=True)
    patch.apply()
    assert patch._PATCHED is True
    assert patch.os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"


def test_apply_leaves_default_when_probe_fails(monkeypatch):
    _wsl(monkeypatch, probe=False)
    patch.apply()
    assert patch._PATCHED is False
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in patch.os.environ


def test_apply_skips_when_disabled(monkeypatch):
    _wsl(monkeypatch, probe=True)
    monkeypatch.setenv("INFERENCE_X_DISABLE_WSL_PIN_MEMORY", "1")
    patch.apply()
    assert patch._PATCHED is False
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in patch.os.environ


def test_apply_noop_off_wsl(monkeypatch):
    _wsl(monkeypatch, probe=True)
    monkeypatch.setattr(patch, "_is_wsl", lambda: False)
    patch.apply()
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in patch.os.environ
