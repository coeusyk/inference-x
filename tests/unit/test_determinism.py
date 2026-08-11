"""Unit tests for SM probe and determinism gating (GPU-free)."""

from __future__ import annotations

import pytest

from inference_x.utils.cuda_env import supports_deterministic_batch_invariant
from inference_x.utils.determinism import (
    DeterminismUnsupportedError,
    batch_invariant_env_enabled,
    ensure_deterministic_mode,
)


@pytest.mark.parametrize(
    ("cap", "ok"),
    [
        ((8, 0), True),
        ((8, 9), True),
        ((9, 0), True),
        ((7, 5), False),
        (None, False),
    ],
)
def test_supports_deterministic_batch_invariant(cap, ok):
    assert supports_deterministic_batch_invariant(cap) is ok


def test_ensure_deterministic_mode_refuses_unsupported(monkeypatch):
    monkeypatch.setattr(
        "inference_x.utils.determinism.probe_compute_capability", lambda: (7, 5)
    )
    with pytest.raises(DeterminismUnsupportedError, match="SM ≥ 8.0"):
        ensure_deterministic_mode(require=True)


def test_ensure_deterministic_mode_quiet_when_not_required(monkeypatch):
    monkeypatch.setattr(
        "inference_x.utils.determinism.probe_compute_capability", lambda: None
    )
    assert ensure_deterministic_mode(require=False) is False


def test_ensure_deterministic_mode_sets_env_on_supported(monkeypatch):
    monkeypatch.setattr(
        "inference_x.utils.determinism.probe_compute_capability", lambda: (8, 9)
    )
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)

    def _fake_init():
        return None

    monkeypatch.setattr(
        "vllm.model_executor.layers.batch_invariant.init_batch_invariance",
        _fake_init,
        raising=False,
    )
    # Import path used inside ensure_deterministic_mode — stub the import target.
    import sys
    from types import ModuleType

    mod = ModuleType("vllm.model_executor.layers.batch_invariant")
    mod.init_batch_invariance = _fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.batch_invariant", mod
    )

    assert ensure_deterministic_mode(require=True) is True
    assert batch_invariant_env_enabled() is True
