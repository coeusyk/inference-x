"""Tests for eager startup initialization."""
import inference_x.api.deps as deps_module
import pytest

from inference_x.api.deps import _build_registry, _build_router, initialize_app
from inference_x.core.settings import get_settings

_REAL_INITIALIZE = initialize_app


@pytest.fixture(autouse=True)
def _restore_real_initialize_app(monkeypatch):
    """Override conftest noop so these tests exercise real startup logic."""
    monkeypatch.setattr(deps_module, "initialize_app", _REAL_INITIALIZE)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Startup tests must bypass conftest's initialize_app patch."""
    get_settings.cache_clear()
    _build_registry.cache_clear()
    _build_router.cache_clear()
    from inference_x.api.deps import _build_engine_pool

    _build_engine_pool.cache_clear()
    yield
    get_settings.cache_clear()
    _build_registry.cache_clear()
    _build_router.cache_clear()
    _build_engine_pool.cache_clear()


def test_initialize_app_fails_on_missing_default_model(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: only-model\n"
        "    engine: vllm\n"
        "    model_path: org/only\n"
    )
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "missing-default")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="not in the registry"):
        _build_router(str(config_dir), "missing-default")


def test_initialize_app_builds_registry_and_router(monkeypatch, tmp_path):
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
    get_settings.cache_clear()

    from inference_x.engines.base import BaseEngine
    from inference_x.engines.pool import EnginePool
    from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamChunk,
)

    class _Stub(BaseEngine):
        async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            raise NotImplementedError

        async def generate_stream(self, request: ChatCompletionRequest):
            yield ChatStreamChunk(content="stub")
            yield ChatStreamChunk(content="", finish_reason="stop")

        def is_healthy(self) -> bool:
            return True

    def _fake_pool_build(config_dir: str, loaded_models: tuple) -> EnginePool:
        return EnginePool({m: _Stub() for m in loaded_models})

    monkeypatch.setattr("inference_x.api.deps._build_engine_pool", _fake_pool_build)

    initialize_app()


def test_initialize_app_loads_multiple_models(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: alpha\n"
        "    engine: vllm\n"
        "    model_path: org/alpha\n"
        "  - name: beta\n"
        "    engine: vllm\n"
        "    model_path: org/beta\n"
    )
    monkeypatch.setenv("INFERENCE_X_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "alpha")
    monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", "alpha,beta")
    get_settings.cache_clear()

    from inference_x.engines.base import BaseEngine
    from inference_x.engines.pool import EnginePool
    from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamChunk,
)

    class _Stub(BaseEngine):
        async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            raise NotImplementedError

        async def generate_stream(self, request: ChatCompletionRequest):
            yield ChatStreamChunk(content="stub")
            yield ChatStreamChunk(content="", finish_reason="stop")

        def is_healthy(self) -> bool:
            return True

    built: dict = {}

    def _fake_pool_build(config_dir: str, loaded_models: tuple) -> EnginePool:
        for m in loaded_models:
            built[m] = _Stub()
        return EnginePool(dict(built))

    monkeypatch.setattr("inference_x.api.deps._build_engine_pool", _fake_pool_build)

    initialize_app()

    assert "alpha" in built
    assert "beta" in built


def test_lifespan_raises_on_startup_failure(monkeypatch):
    def _fail_init():
        raise RuntimeError("engine load failed")

    monkeypatch.setattr(deps_module, "initialize_app", _fail_init)

    from fastapi.testclient import TestClient

    from inference_x.api.main import app

    with pytest.raises(RuntimeError, match="engine load failed"):
        with TestClient(app):
            pass
