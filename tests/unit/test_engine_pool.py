"""Unit tests for EnginePool."""
import pytest

from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class _StubEngine(BaseEngine):
    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    async def generate_stream(self, request: ChatCompletionRequest):
        yield "stub"

    def is_healthy(self) -> bool:
        return self._healthy


def _pool(*names: str, healthy: bool = True) -> EnginePool:
    return EnginePool({n: _StubEngine(healthy=healthy) for n in names})


class TestEnginePool:
    def test_requires_at_least_one_engine(self):
        with pytest.raises(ValueError):
            EnginePool({})

    def test_get_returns_correct_engine(self):
        engine_a = _StubEngine()
        engine_b = _StubEngine()
        pool = EnginePool({"a": engine_a, "b": engine_b})
        assert pool.get("a") is engine_a
        assert pool.get("b") is engine_b

    def test_get_raises_for_unknown_model(self):
        pool = _pool("alpha")
        with pytest.raises(ValueError, match="not loaded"):
            pool.get("beta")

    def test_loaded_models_returns_all_names(self):
        pool = EnginePool({"x": _StubEngine(), "y": _StubEngine(), "z": _StubEngine()})
        assert pool.loaded_models() == ["x", "y", "z"]

    def test_all_healthy_true_when_all_ok(self):
        pool = _pool("a", "b", "c", healthy=True)
        assert pool.all_healthy() is True

    def test_all_healthy_false_when_any_unhealthy(self):
        pool = EnginePool({"ok": _StubEngine(healthy=True), "bad": _StubEngine(healthy=False)})
        assert pool.all_healthy() is False

    def test_health_status_maps_names_to_state(self):
        pool = EnginePool({"ok": _StubEngine(True), "bad": _StubEngine(False)})
        status = pool.health_status()
        assert status["ok"] == "ok"
        assert status["bad"] == "unavailable"

    def test_single_entry_pool_mirrors_single_engine_behaviour(self):
        engine = _StubEngine(healthy=True)
        pool = EnginePool({"solo": engine})
        assert pool.get("solo") is engine
        assert pool.all_healthy() is True
        assert pool.loaded_models() == ["solo"]
