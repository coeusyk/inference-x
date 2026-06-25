"""Unit tests for routing layer: TaskRouter and selection policies."""
import pytest

from inference_x.routing.policies import DefaultModelPolicy, ExplicitModelPolicy
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage
from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry(*names: str) -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=n, model_path=f"org/{n}") for n in names])


def _req(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model, messages=[ChatMessage(role="user", content="hi")]
    )


# ---------------------------------------------------------------------------
# ExplicitModelPolicy
# ---------------------------------------------------------------------------

class TestExplicitModelPolicy:
    def test_returns_model_name_when_registered(self):
        policy = ExplicitModelPolicy(_registry("a", "b"))
        assert policy.apply(_req("a")) == "a"

    def test_returns_none_when_not_registered(self):
        policy = ExplicitModelPolicy(_registry("a"))
        assert policy.apply(_req("unknown")) is None


# ---------------------------------------------------------------------------
# DefaultModelPolicy
# ---------------------------------------------------------------------------

class TestDefaultModelPolicy:
    def test_always_returns_default(self):
        policy = DefaultModelPolicy("b", _registry("a", "b"))
        assert policy.apply(_req("anything")) == "b"

    def test_default_not_in_registry_raises(self):
        with pytest.raises(ValueError, match="not in the registry"):
            DefaultModelPolicy("missing", _registry("a"))


# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------

class TestTaskRouter:
    def test_explicit_model_wins(self):
        router = TaskRouter(_registry("a", "b"), default_model="b")
        assert router.select(_req("a")) == "a"

    def test_falls_back_to_default_for_unknown_model(self):
        router = TaskRouter(_registry("a", "b"), default_model="b")
        assert router.select(_req("nonexistent")) == "b"

    def test_default_is_used_when_client_requests_it_explicitly(self):
        router = TaskRouter(_registry("a", "b"), default_model="b")
        assert router.select(_req("b")) == "b"

    def test_invalid_default_raises_at_construction(self):
        with pytest.raises(ValueError):
            TaskRouter(_registry("a"), default_model="missing")

    def test_config_driven_default(self):
        """Changing the default_model changes routing for unregistered requests."""
        reg = _registry("fast", "accurate")
        r1 = TaskRouter(reg, default_model="fast")
        r2 = TaskRouter(reg, default_model="accurate")
        req = _req("unknown")
        assert r1.select(req) == "fast"
        assert r2.select(req) == "accurate"
