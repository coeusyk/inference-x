"""Tests for WSL pin_memory platform patch."""

from __future__ import annotations

import inference_x.utils.vllm_platform_patch as patch


def test_apply_is_idempotent(monkeypatch):
    monkeypatch.setattr(patch, "_PATCHED", False)
    monkeypatch.setenv("INFERENCE_X_VLLM_PATCH_APPLIED", "1")
    patch.apply()
    assert patch._PATCHED is True


def test_apply_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(patch, "_PATCHED", False)
    monkeypatch.delenv("INFERENCE_X_VLLM_PATCH_APPLIED", raising=False)
    monkeypatch.setenv("INFERENCE_X_DISABLE_WSL_PIN_MEMORY", "1")
    monkeypatch.setattr(patch, "_is_wsl", lambda: True)
    patch.apply()
    assert patch._PATCHED is False
