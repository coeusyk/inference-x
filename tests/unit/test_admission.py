"""Unit tests for routing/admission.py (AdmissionController)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from inference_x.routing.admission import (
    AdmissionController,
    ContextTooLongError,
    EngineSaturatedError,
)
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage
from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry


@dataclass
class _FakeTier:
    max_model_len_cap: int


class _FakeEngine:
    """Engine stub exposing the two optional attributes AdmissionController reads."""

    def __init__(self, prompt_tokens: int = 10, kv_capacity_tokens: Optional[int] = None) -> None:
        self._prompt_tokens = prompt_tokens
        self.kv_capacity_tokens = kv_capacity_tokens

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        return self._prompt_tokens


def _registry(**overrides) -> ModelRegistry:
    return ModelRegistry([ModelEntry(name="m", model_path="test/m", **overrides)])


def _req(**overrides) -> ChatCompletionRequest:
    defaults = dict(model="m", messages=[ChatMessage(role="user", content="hi")])
    defaults.update(overrides)
    return ChatCompletionRequest(**defaults)


class TestContextCeiling:
    def test_no_tier_uses_default_4096_cap(self):
        controller = AdmissionController(_registry())
        result = controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 100

    def test_tier_cap_is_used_when_no_model_max_model_len(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=64))
        # prompt(10) + requested(100) > 64 -> clamp to 54 (interactive default)
        result = controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 54

    def test_model_max_model_len_further_restricts_tier_cap(self):
        controller = AdmissionController(
            _registry(max_model_len=32), tier=_FakeTier(max_model_len_cap=4096)
        )
        result = controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 22  # 32 - 10

    def test_request_max_context_tokens_further_restricts(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096))
        result = controller.admit(
            "m", _req(max_tokens=100, max_context_tokens=30), _FakeEngine(prompt_tokens=10)
        )
        assert result.effective_max_tokens == 20  # 30 - 10


class TestPromptTooLong:
    def test_prompt_alone_exceeding_ceiling_raises_400_class(self):
        controller = AdmissionController(_registry(max_model_len=8))
        with pytest.raises(ContextTooLongError, match="Prompt is 10 tokens"):
            controller.admit("m", _req(), _FakeEngine(prompt_tokens=10))

    def test_context_too_long_is_a_value_error(self):
        """ContextTooLongError must subclass ValueError to hit the sanitized 400 handler."""
        assert issubclass(ContextTooLongError, ValueError)


class TestClampVsReject:
    def test_interactive_clamps_when_room_is_sufficient(self):
        controller = AdmissionController(_registry(max_model_len=100))
        result = controller.admit(
            "m", _req(max_tokens=200, priority="interactive"), _FakeEngine(prompt_tokens=10)
        )
        assert result.effective_max_tokens == 90

    def test_batch_rejects_instead_of_clamping(self):
        controller = AdmissionController(_registry(max_model_len=100))
        with pytest.raises(ContextTooLongError):
            controller.admit(
                "m", _req(max_tokens=200, priority="batch"), _FakeEngine(prompt_tokens=10)
            )

    def test_interactive_rejects_when_clamped_room_below_minimum(self):
        """Room < _MIN_CLAMPED_OUTPUT_TOKENS (16) is treated as a rejection, not a
        near-useless clamp to a handful of tokens."""
        controller = AdmissionController(_registry(max_model_len=15))
        with pytest.raises(ContextTooLongError):
            controller.admit(
                "m", _req(max_tokens=200, priority="interactive"), _FakeEngine(prompt_tokens=10)
            )


class TestKvSaturation:
    def test_no_capacity_reported_never_blocks(self):
        controller = AdmissionController(_registry())
        result = controller.admit(
            "m", _req(max_tokens=4000), _FakeEngine(prompt_tokens=10, kv_capacity_tokens=None)
        )
        assert result.effective_max_tokens == 4000

    def test_batch_rejected_with_429_class_when_saturated(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        # Fill the reservation close to the (0.9 * 100 = 90) budget.
        controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        with pytest.raises(EngineSaturatedError):
            controller.admit("m", _req(max_tokens=50, priority="batch"), engine)

    def test_interactive_clamps_under_kv_pressure(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        controller.admit("m", _req(max_tokens=60, priority="batch"), engine)
        # budget=90, reserved=70, prompt=10 -> available=10 tokens, below the 16-token
        # minimum, so even interactive gets rejected outright rather than a token clamp.
        with pytest.raises(EngineSaturatedError):
            controller.admit("m", _req(max_tokens=50, priority="interactive"), engine)

    def test_release_frees_reserved_tokens_for_next_admission(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        first = controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        controller.release("m", first.reserved_tokens)
        # Without release() this would raise EngineSaturatedError (see test above).
        second = controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        assert second.effective_max_tokens == 70

    def test_reservations_are_tracked_per_model(self):
        registry = ModelRegistry(
            [
                ModelEntry(name="a", model_path="test/a"),
                ModelEntry(name="b", model_path="test/b"),
            ]
        )
        controller = AdmissionController(registry)
        engine_a = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        engine_b = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        controller.admit("a", _req(model="a", max_tokens=80, priority="batch"), engine_a)
        # Model b's budget is untouched by a's reservation.
        result = controller.admit("b", _req(model="b", max_tokens=80, priority="batch"), engine_b)
        assert result.effective_max_tokens == 80


class TestPromptTokenFallback:
    def test_falls_back_to_chars_over_4_when_engine_has_no_tokenizer(self):
        controller = AdmissionController(_registry(max_model_len=4))

        class _NoTokenizerEngine:
            pass

        # "hi" is 2 chars -> max(1, 2 // 4) == 1 prompt token, well under the cap.
        result = controller.admit("m", _req(max_tokens=1), _NoTokenizerEngine())
        assert result.effective_max_tokens == 1
