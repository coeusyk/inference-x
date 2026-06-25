"""Tests for HuggingFace preflight helpers in vllm_engine."""
from __future__ import annotations

from unittest import mock

import pytest

from inference_x.engines.vllm_engine import (
    _is_hf_hub_model_path,
    _map_vllm_init_error,
    preflight_hf_access,
    resolve_hf_token,
)


def _hf_http_error(exc_cls: type[Exception], message: str = "error") -> Exception:
    """Build huggingface_hub HTTP errors (require a response kwarg)."""
    response = mock.Mock(status_code=403, headers={})
    return exc_cls(message, response=response)


def test_resolve_hf_token_prefers_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "other")
    assert resolve_hf_token() == "hf_test"


def test_resolve_hf_token_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert resolve_hf_token() is None


def test_is_hf_hub_model_path_detects_hub_ids():
    assert _is_hf_hub_model_path("meta-llama/Meta-Llama-3-8B-Instruct") is True
    assert _is_hf_hub_model_path("Qwen/Qwen2.5-0.5B-Instruct") is True


def test_is_hf_hub_model_path_skips_local_paths():
    assert _is_hf_hub_model_path("/home/user/models/llama") is False
    assert _is_hf_hub_model_path("./models/llama") is False
    assert _is_hf_hub_model_path("../models/llama") is False
    assert _is_hf_hub_model_path("~/models/llama") is False


def test_preflight_skips_local_paths():
    preflight_hf_access("local", "/var/models/my-llm", token=None)


def test_preflight_no_probe_for_public_ungated_hub_model():
    info = mock.Mock(gated=False, private=False)
    with mock.patch("huggingface_hub.model_info", return_value=info) as mi:
        with mock.patch("huggingface_hub.hf_hub_download") as dl:
            preflight_hf_access("qwen", "Qwen/Qwen2.5-0.5B-Instruct", token=None)
    mi.assert_called_once()
    dl.assert_not_called()


def test_preflight_gated_without_token_fails_fast():
    from huggingface_hub.errors import GatedRepoError

    info = mock.Mock(gated="manual", private=False)
    with mock.patch("huggingface_hub.model_info", return_value=info):
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=_hf_http_error(GatedRepoError, "gated"),
        ):
            with pytest.raises(RuntimeError, match="gated on HuggingFace"):
                preflight_hf_access(
                    "llama3-8b",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    token=None,
                )


def test_preflight_gated_with_token_not_approved():
    from huggingface_hub.errors import GatedRepoError

    info = mock.Mock(gated="manual", private=False)
    with mock.patch("huggingface_hub.model_info", return_value=info):
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=_hf_http_error(GatedRepoError, "gated"),
        ):
            with pytest.raises(RuntimeError, match="not yet approved"):
                preflight_hf_access(
                    "llama3-8b",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    token="hf_test",
                )


def test_preflight_gated_with_valid_access():
    info = mock.Mock(gated="manual", private=False)
    with mock.patch("huggingface_hub.model_info", return_value=info):
        with mock.patch("huggingface_hub.hf_hub_download", return_value="/tmp/config.json"):
            preflight_hf_access(
                "llama3-8b",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                token="hf_test",
            )


def test_preflight_repo_not_found():
    from huggingface_hub.errors import RepositoryNotFoundError

    with mock.patch(
        "huggingface_hub.model_info",
        side_effect=_hf_http_error(RepositoryNotFoundError, "missing"),
    ):
        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            preflight_hf_access("bad", "no-such/org-model", token=None)


def test_map_vllm_init_error_detects_insufficient_free_vram():
    exc = ValueError(
        "Free memory on device cuda:0 (6.93/8.0 GiB) on startup is less than "
        "desired GPU memory utilization (0.9, 7.2 GiB)."
    )
    err = _map_vllm_init_error("llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct", exc)
    assert "GPU memory insufficient" in str(err)


def test_check_vram_budget_raises_when_free_below_requested():
    from inference_x.engines.vllm_engine import _check_vram_budget

    with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
        _check_vram_budget(
            "llama3-8b",
            0.9,
            {"cuda_available": True, "free_gib": 6.93, "total_gib": 8.0},
        )


def test_check_vram_budget_passes_when_free_meets_requested():
    from inference_x.engines.vllm_engine import _check_vram_budget

    _check_vram_budget(
        "llama3-8b",
        0.85,
        {"cuda_available": True, "free_gib": 6.93, "total_gib": 8.0},
    )


def test_map_vllm_init_error_detects_gated_repo_fallback():
    exc = OSError(
        "You are trying to access a gated repo. "
        "Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted."
    )
    err = _map_vllm_init_error("llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct", exc)
    assert "gated on HuggingFace" in str(err)


def test_vllm_engine_preflight_before_vllm_load(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    from inference_x.engines.vllm_engine import VLLMEngine

    with mock.patch(
        "inference_x.engines.vllm_engine.preflight_hf_access",
        side_effect=RuntimeError("preflight blocked"),
    ) as preflight:
        with pytest.raises(RuntimeError, match="preflight blocked"):
            VLLMEngine(
                {
                    "name": "llama3-8b",
                    "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
                }
            )
    preflight.assert_called_once()
