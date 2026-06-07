"""Unit tests for chat schemas."""
import pytest
from pydantic import ValidationError

from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatMessage,
)


class TestChatMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="unknown", content="hi")


class TestChatCompletionRequest:
    def test_minimal_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert req.model == "test-model"
        assert req.temperature == 0.7
        assert req.max_tokens == 512
        assert req.stream is False

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=3.0,
            )

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=0,
            )


class TestChatCompletionResponse:
    def _make_response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        )

    def test_response_structure(self):
        resp = self._make_response()
        assert resp.object == "chat.completion"
        assert resp.model == "test-model"
        assert len(resp.choices) == 1
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.total_tokens == 8

    def test_auto_id_generated(self):
        r1 = self._make_response()
        r2 = self._make_response()
        assert r1.id.startswith("chatcmpl-")
        assert r1.id != r2.id

    def test_serializes_to_dict(self):
        resp = self._make_response()
        data = resp.model_dump()
        assert "choices" in data
        assert "usage" in data
