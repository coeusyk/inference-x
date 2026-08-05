"""Unit tests for routing/admission.py (AdmissionController)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from inference_x.engines.base import BaseEngine
from inference_x.routing.admission import (
    AdmissionController,
    ContextTooLongError,
    EngineSaturatedError,
    StrictModeViolationError,
)
from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage
from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry


@dataclass
class _FakeTier:
    max_model_len_cap: int
    max_num_seqs: int = 1000  # high enough to not trip the seq-concurrency gate by default


class _FakeEngine(BaseEngine):
    """Engine stub: the declared count_prompt_tokens plus the optional KV attribute.

    Subclasses BaseEngine because admit() takes one — count_prompt_tokens is a
    declared capability since OS-4, not a getattr probe (DEC-047 §3).
    """

    def __init__(self, prompt_tokens: int = 10, kv_capacity_tokens: Optional[int] = None) -> None:
        self._prompt_tokens = prompt_tokens
        self.kv_capacity_tokens = kv_capacity_tokens

    async def generate(self, request):  # pragma: no cover - admission never generates
        raise AssertionError("not used")

    async def generate_stream(self, request):  # pragma: no cover - never used
        raise AssertionError("not used")
        yield

    def is_healthy(self) -> bool:
        return True

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        return self._prompt_tokens


class _NoTokenizerEngine(_FakeEngine):
    """An engine with no tokenizer to ask — the BaseEngine default applies."""

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int | None:
        return None


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

        # "hi" is 2 chars -> max(1, 2 // 4) == 1 prompt token, well under the cap.
        result = controller.admit("m", _req(max_tokens=1), _NoTokenizerEngine())
        assert result.effective_max_tokens == 1

    def test_the_estimate_is_reported_rather_than_absorbed(self):
        controller = AdmissionController(_registry(max_model_len=4))
        result = controller.admit("m", _req(max_tokens=1), _NoTokenizerEngine())
        estimated = [w for w in result.warnings if w.code == "prompt_tokens_estimated"]
        assert len(estimated) == 1
        assert estimated[0].type == "degraded"
        assert estimated[0].field == "messages"


class TestTypedDegradation:
    """§9 C.8: a skipped gate must be distinguishable from a gate that passed.

    Every one of these conditions must still ADMIT. Making degradation visible
    is the change; making it reject would turn admission fail-closed, which
    DEC-047 §4 forbids for this milestone.
    """

    def test_no_tier_reports_the_skipped_sequence_gate(self):
        controller = AdmissionController(_registry())
        result = controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        assert "sequence_gate_skipped" in {w.code for w in result.warnings}

    def test_no_kv_capacity_reports_the_skipped_kv_gate(self):
        controller = AdmissionController(_registry())
        result = controller.admit(
            "m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1, kv_capacity_tokens=None)
        )
        assert "kv_gate_skipped" in {w.code for w in result.warnings}

    @pytest.mark.parametrize("strict", [False, True])
    def test_no_degraded_condition_ever_rejects(self, strict: bool):
        """DEC-047 §4 — fail open, under either strict value."""
        controller = AdmissionController(_registry(max_model_len=4096))
        result = controller.admit("m", _req(max_tokens=10, strict=strict), _NoTokenizerEngine())
        assert result.effective_max_tokens == 10
        assert {w.type for w in result.warnings} == {"degraded"}

    def test_a_clean_request_carries_no_warnings(self):
        """The empty case. A warning path that stops firing fails silently."""
        controller = AdmissionController(
            _registry(max_model_len=4096), tier=_FakeTier(max_model_len_cap=4096)
        )
        result = controller.admit(
            "m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1, kv_capacity_tokens=100_000)
        )
        assert result.warnings == ()


class TestStrictMode:
    """DEC-052: strict may only convert a substitution into a rejection."""

    def test_strict_rejects_where_the_default_clamps_on_context(self):
        controller = AdmissionController(_registry(max_model_len=4096))
        engine = _FakeEngine(prompt_tokens=10)
        req = dict(max_tokens=4000, max_context_tokens=200)

        lenient = controller.admit("m", _req(**req), engine)
        assert lenient.effective_max_tokens == 190
        assert "max_tokens_clamped_to_context" in {w.code for w in lenient.warnings}

        with pytest.raises(StrictModeViolationError):
            controller.admit("m", _req(strict=True, **req), engine)

    def test_strict_rejects_where_the_default_clamps_on_kv_budget(self):
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)

        # A fresh controller per call: admit() reserves against the KV tracker, so
        # reusing one would saturate the pool and mask the strict check.
        lenient = AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=4000), engine
        )
        assert "max_tokens_clamped_to_kv_budget" in {w.code for w in lenient.warnings}

        with pytest.raises(StrictModeViolationError):
            AdmissionController(_registry(max_model_len=4096)).admit(
                "m", _req(max_tokens=4000, strict=True), engine
            )

    def test_strict_rejection_set_equals_the_substituted_warning_set(self):
        """DEC-052 §2 — one predicate, two outcomes, enumerated not sampled.

        Every condition that emits a `substituted` warning by default must raise
        under strict, and nothing else may. Sampling two cases would pass while
        the two sets drifted apart.
        """
        cases = {
            "max_tokens_clamped_to_context": (
                dict(max_tokens=4000, max_context_tokens=200),
                _FakeEngine(prompt_tokens=10),
            ),
            "max_tokens_clamped_to_kv_budget": (
                dict(max_tokens=4000),
                _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100),
            ),
        }

        warns_by_default: set[str] = set()
        rejects_under_strict: set[str] = set()

        for code, (kwargs, engine) in cases.items():
            controller = AdmissionController(_registry(max_model_len=4096))
            result = controller.admit("m", _req(**kwargs), engine)
            warns_by_default |= {
                w.code for w in result.warnings if w.type == "substituted"
            }

            controller = AdmissionController(_registry(max_model_len=4096))
            try:
                controller.admit("m", _req(strict=True, **kwargs), engine)
            except StrictModeViolationError:
                rejects_under_strict.add(code)

        assert warns_by_default == rejects_under_strict == set(cases)

    def test_strict_does_not_change_an_admitted_result(self):
        """Accepted under both modes -> identical AdmissionResult numbers."""
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100_000)
        lenient = AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=100), engine
        )
        strict = AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=100, strict=True), engine
        )
        assert lenient.effective_max_tokens == strict.effective_max_tokens
        assert lenient.reserved_tokens == strict.reserved_tokens


class TestSequenceConcurrencyGate:
    def test_no_tier_never_blocks_on_sequence_count(self):
        """No tier resolved -> gate is skipped entirely (fail open)."""
        controller = AdmissionController(_registry())
        for _ in range(50):
            controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        # No exception raised for any of the 50 concurrent (never-released) admissions.

    def test_batch_rejected_at_tier_max_num_seqs(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=2))
        controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))

    def test_interactive_also_rejected_at_tier_max_num_seqs(self):
        """Unlike the KV-token gate, there is no clamp path for a sequence slot —
        interactive priority is rejected too when the ceiling is hit."""
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1))
        controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    def test_model_max_num_seqs_override_further_restricts_tier(self):
        controller = AdmissionController(
            _registry(max_num_seqs=1), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=8)
        )
        controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    def test_release_frees_a_sequence_slot(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1))
        result = controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        controller.release("m", result.reserved_tokens)
        # Slot freed -> a second admission succeeds instead of raising.
        controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    def test_sequence_counts_are_tracked_per_model(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1))
        controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        # A different model has its own independent slot count.
        controller._registry = ModelRegistry(
            [ModelEntry(name="m", model_path="test/m"), ModelEntry(name="other", model_path="test/other")]
        )
        controller.admit("other", _req(model="other", max_tokens=10), _FakeEngine(prompt_tokens=1))

    def test_a_request_rejected_by_context_gate_does_not_consume_a_sequence_slot(self):
        """admit() must be atomic: a later-gate rejection must not leak a slot that
        release() will never be called for (ChatService only calls release() when
        admit() succeeds)."""
        controller = AdmissionController(
            _registry(max_model_len=4), tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1)
        )
        with pytest.raises(ContextTooLongError):
            controller.admit("m", _req(), _FakeEngine(prompt_tokens=10))
        # Slot was never consumed -> a normal admission still succeeds.
        controller.admit("m", _req(max_tokens=1), _FakeEngine(prompt_tokens=1))
