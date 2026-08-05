"""Tests for graceful vLLM engine shutdown."""

from functools import lru_cache
from unittest.mock import MagicMock

import inference_x.api.deps as deps_module
from inference_x.api.deps import shutdown_app
from inference_x.engines.pool import EnginePool
from inference_x.engines.vllm_engine import VLLMEngine


def test_vllm_engine_shutdown_stops_engine_core():
    """AsyncLLM.shutdown() is called directly (migrate-async-llm-engine) —
    no repo-side llm_engine.engine_core indirection, unlike the offline LLM
    class this replaced."""
    engine = object.__new__(VLLMEngine)
    engine._model_name = "test-model"
    engine._healthy = True

    mock_llm = MagicMock()
    engine._llm = mock_llm

    engine.shutdown()

    mock_llm.shutdown.assert_called_once_with()
    assert engine._llm is None
    assert engine.is_healthy() is False


def test_engine_pool_shutdown_calls_each_engine():
    first = MagicMock()
    second = MagicMock()
    pool = EnginePool({"a": first, "b": second})

    pool.shutdown()

    first.shutdown.assert_called_once_with()
    second.shutdown.assert_called_once_with()


def test_shutdown_app_shuts_down_cached_pool(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: alpha\n"
        "    engine: vllm\n"
        "    model_path: org/alpha\n"
    )
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "alpha")
    monkeypatch.delenv("INFERENCE_X_LOADED_MODELS", raising=False)

    from inference_x.core.settings import get_settings

    get_settings.cache_clear()
    deps_module._build_registry.cache_clear()
    deps_module._build_router.cache_clear()
    deps_module._build_engine_pool.cache_clear()

    pool = MagicMock()
    monkeypatch.setattr(
        deps_module,
        "_build_engine_pool",
        lru_cache(maxsize=1)(lambda config_dir, loaded_models: pool),
    )

    settings = get_settings()
    deps_module._build_engine_pool(settings.config_dir, tuple(settings.loaded_models))
    shutdown_app()

    pool.shutdown.assert_called_once_with()
    assert deps_module._build_engine_pool.cache_info().currsize == 0
