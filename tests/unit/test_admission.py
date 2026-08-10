"""Unit tests for routing/admission.py (AdmissionController)."""
from __future__ import annotations

import asyncio
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
    async def test_no_tier_uses_default_4096_cap(self):
        controller = AdmissionController(_registry())
        result = await controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 100

    async def test_tier_cap_is_used_when_no_model_max_model_len(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=64))
        # prompt(10) + requested(100) > 64 -> clamp to 54 (interactive default)
        result = await controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 54

    async def test_model_max_model_len_further_restricts_tier_cap(self):
        controller = AdmissionController(
            _registry(max_model_len=32), tier=_FakeTier(max_model_len_cap=4096)
        )
        result = await controller.admit("m", _req(max_tokens=100), _FakeEngine(prompt_tokens=10))
        assert result.effective_max_tokens == 22  # 32 - 10

    async def test_request_max_context_tokens_further_restricts(self):
        controller = AdmissionController(_registry(), tier=_FakeTier(max_model_len_cap=4096))
        result = await controller.admit(
            "m", _req(max_tokens=100, max_context_tokens=30), _FakeEngine(prompt_tokens=10)
        )
        assert result.effective_max_tokens == 20  # 30 - 10


class TestPromptTooLong:
    async def test_prompt_alone_exceeding_ceiling_raises_400_class(self):
        controller = AdmissionController(_registry(max_model_len=8))
        with pytest.raises(ContextTooLongError, match="Prompt is 10 tokens"):
            await controller.admit("m", _req(), _FakeEngine(prompt_tokens=10))

    def test_context_too_long_is_a_value_error(self):
        """ContextTooLongError must subclass ValueError to hit the sanitized 400 handler."""
        assert issubclass(ContextTooLongError, ValueError)


class TestClampVsReject:
    async def test_interactive_clamps_when_room_is_sufficient(self):
        controller = AdmissionController(_registry(max_model_len=100))
        result = await controller.admit(
            "m", _req(max_tokens=200, priority="interactive"), _FakeEngine(prompt_tokens=10)
        )
        assert result.effective_max_tokens == 90

    async def test_batch_rejects_instead_of_clamping(self):
        controller = AdmissionController(_registry(max_model_len=100))
        with pytest.raises(ContextTooLongError):
            await controller.admit(
                "m", _req(max_tokens=200, priority="batch"), _FakeEngine(prompt_tokens=10)
            )

    async def test_interactive_rejects_when_clamped_room_below_minimum(self):
        """Room < _MIN_CLAMPED_OUTPUT_TOKENS (16) is treated as a rejection, not a
        near-useless clamp to a handful of tokens."""
        controller = AdmissionController(_registry(max_model_len=15))
        with pytest.raises(ContextTooLongError):
            await controller.admit(
                "m", _req(max_tokens=200, priority="interactive"), _FakeEngine(prompt_tokens=10)
            )


class TestKvSaturation:
    async def test_no_capacity_reported_never_blocks(self):
        controller = AdmissionController(_registry())
        result = await controller.admit(
            "m", _req(max_tokens=4000), _FakeEngine(prompt_tokens=10, kv_capacity_tokens=None)
        )
        assert result.effective_max_tokens == 4000

    async def test_batch_rejected_with_429_class_when_saturated(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        # Fill the reservation close to the (0.9 * 100 = 90) budget.
        await controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        with pytest.raises(EngineSaturatedError):
            await controller.admit("m", _req(max_tokens=50, priority="batch"), engine)

    async def test_interactive_clamps_under_kv_pressure(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        await controller.admit("m", _req(max_tokens=60, priority="batch"), engine)
        # budget=90, reserved=70, prompt=10 -> available=10 tokens, below the 16-token
        # minimum, so even interactive gets rejected outright rather than a token clamp.
        with pytest.raises(EngineSaturatedError):
            await controller.admit("m", _req(max_tokens=50, priority="interactive"), engine)

    async def test_release_frees_reserved_tokens_for_next_admission(self):
        controller = AdmissionController(_registry())
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        first = await controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        controller.release("m", first.reserved_tokens)
        # Without release() this would raise EngineSaturatedError (see test above).
        second = await controller.admit("m", _req(max_tokens=70, priority="batch"), engine)
        assert second.effective_max_tokens == 70

    async def test_reservations_are_tracked_per_model(self):
        registry = ModelRegistry(
            [
                ModelEntry(name="a", model_path="test/a"),
                ModelEntry(name="b", model_path="test/b"),
            ]
        )
        controller = AdmissionController(registry)
        engine_a = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        engine_b = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)
        await controller.admit("a", _req(model="a", max_tokens=80, priority="batch"), engine_a)
        # Model b's budget is untouched by a's reservation.
        result = await controller.admit(
            "b", _req(model="b", max_tokens=80, priority="batch"), engine_b
        )
        assert result.effective_max_tokens == 80


class TestPromptTokenFallback:
    async def test_falls_back_to_chars_over_4_when_engine_has_no_tokenizer(self):
        controller = AdmissionController(_registry(max_model_len=4))

        # "hi" is 2 chars -> max(1, 2 // 4) == 1 prompt token, well under the cap.
        result = await controller.admit("m", _req(max_tokens=1), _NoTokenizerEngine())
        assert result.effective_max_tokens == 1

    async def test_the_estimate_is_reported_rather_than_absorbed(self):
        controller = AdmissionController(_registry(max_model_len=4))
        result = await controller.admit("m", _req(max_tokens=1), _NoTokenizerEngine())
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

    async def test_no_tier_reports_the_skipped_sequence_gate(self):
        controller = AdmissionController(_registry())
        result = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        assert "sequence_gate_skipped" in {w.code for w in result.warnings}

    async def test_no_kv_capacity_reports_the_skipped_kv_gate(self):
        controller = AdmissionController(_registry())
        result = await controller.admit(
            "m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1, kv_capacity_tokens=None)
        )
        assert "kv_gate_skipped" in {w.code for w in result.warnings}

    @pytest.mark.parametrize("strict", [False, True])
    async def test_no_degraded_condition_ever_rejects(self, strict: bool):
        """DEC-047 §4 — fail open, under either strict value."""
        controller = AdmissionController(_registry(max_model_len=4096))
        result = await controller.admit(
            "m", _req(max_tokens=10, strict=strict), _NoTokenizerEngine()
        )
        assert result.effective_max_tokens == 10
        assert {w.type for w in result.warnings} == {"degraded"}

    async def test_a_clean_request_carries_no_warnings(self):
        """The empty case. A warning path that stops firing fails silently."""
        controller = AdmissionController(
            _registry(max_model_len=4096), tier=_FakeTier(max_model_len_cap=4096)
        )
        result = await controller.admit(
            "m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1, kv_capacity_tokens=100_000)
        )
        assert result.warnings == ()


class TestStrictMode:
    """DEC-052: strict may only convert a substitution into a rejection."""

    async def test_strict_rejects_where_the_default_clamps_on_context(self):
        controller = AdmissionController(_registry(max_model_len=4096))
        engine = _FakeEngine(prompt_tokens=10)
        req = dict(max_tokens=4000, max_context_tokens=200)

        lenient = await controller.admit("m", _req(**req), engine)
        assert lenient.effective_max_tokens == 190
        assert "max_tokens_clamped_to_context" in {w.code for w in lenient.warnings}

        with pytest.raises(StrictModeViolationError):
            await controller.admit("m", _req(strict=True, **req), engine)

    async def test_strict_rejects_where_the_default_clamps_on_kv_budget(self):
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100)

        # A fresh controller per call: admit() reserves against the KV tracker, so
        # reusing one would saturate the pool and mask the strict check.
        lenient = await AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=4000), engine
        )
        assert "max_tokens_clamped_to_kv_budget" in {w.code for w in lenient.warnings}

        with pytest.raises(StrictModeViolationError):
            await AdmissionController(_registry(max_model_len=4096)).admit(
                "m", _req(max_tokens=4000, strict=True), engine
            )

    async def test_strict_rejection_set_equals_the_substituted_warning_set(self):
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
            result = await controller.admit("m", _req(**kwargs), engine)
            warns_by_default |= {
                w.code for w in result.warnings if w.type == "substituted"
            }

            controller = AdmissionController(_registry(max_model_len=4096))
            try:
                await controller.admit("m", _req(strict=True, **kwargs), engine)
            except StrictModeViolationError:
                rejects_under_strict.add(code)

        assert warns_by_default == rejects_under_strict == set(cases)

    async def test_strict_does_not_change_an_admitted_result(self):
        """Accepted under both modes -> identical AdmissionResult numbers."""
        engine = _FakeEngine(prompt_tokens=10, kv_capacity_tokens=100_000)
        lenient = await AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=100), engine
        )
        strict = await AdmissionController(_registry(max_model_len=4096)).admit(
            "m", _req(max_tokens=100, strict=True), engine
        )
        assert lenient.effective_max_tokens == strict.effective_max_tokens
        assert lenient.reserved_tokens == strict.reserved_tokens


_FAST_WAIT_S = 0.05  # keeps bounded-wait tests fast without exercising the real default


class TestSequenceConcurrencyGate:
    async def test_no_tier_never_blocks_on_sequence_count(self):
        """No tier resolved -> gate is skipped entirely (fail open)."""
        controller = AdmissionController(_registry())
        for _ in range(50):
            await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        # No exception raised for any of the 50 concurrent (never-released) admissions.

    async def test_batch_rejected_at_tier_max_num_seqs(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=2),
            admission_wait_s=_FAST_WAIT_S,
            batch_admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            await controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )

    async def test_interactive_also_rejected_at_tier_max_num_seqs(self):
        """Unlike the KV-token gate, there is no clamp path for a sequence slot —
        interactive priority is rejected too when the ceiling stays full past the wait."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_model_max_num_seqs_override_further_restricts_tier(self):
        controller = AdmissionController(
            _registry(max_num_seqs=1),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=8),
            admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_release_frees_a_sequence_slot(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        result = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        controller.release("m", result.reserved_tokens)
        # Slot freed -> a second admission succeeds instead of raising.
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_sequence_counts_are_tracked_per_model(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        # A different model has its own independent slot count.
        controller._registry = ModelRegistry(
            [ModelEntry(name="m", model_path="test/m"), ModelEntry(name="other", model_path="test/other")]
        )
        await controller.admit("other", _req(model="other", max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_a_request_rejected_by_context_gate_does_not_consume_a_sequence_slot(self):
        """admit() must be atomic: a later-gate rejection must not leak a slot that
        release() will never be called for (ChatService only calls release() when
        admit() succeeds)."""
        controller = AdmissionController(
            _registry(max_model_len=4),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        with pytest.raises(ContextTooLongError):
            await controller.admit("m", _req(), _FakeEngine(prompt_tokens=10))
        # Slot was never consumed -> a normal admission still succeeds.
        await controller.admit("m", _req(max_tokens=1), _FakeEngine(prompt_tokens=1))

    async def test_double_release_raises_value_error_instead_of_inflating_capacity(self):
        """BoundedSemaphore, not plain Semaphore: an over-release must fail loudly
        rather than silently letting capacity exceed max_num_seqs (design.md
        "Semaphore lifecycle analysis" — the defensive reason BoundedSemaphore
        was chosen over plain Semaphore)."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        result = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        controller.release("m", result.reserved_tokens)
        with pytest.raises(ValueError):
            controller.release("m", result.reserved_tokens)

    async def test_stray_release_below_capacity_is_not_caught_by_bounded_semaphore(self):
        """Document BoundedSemaphore's actual over-release protection boundary:
        it only raises when the release would push the internal count to or
        past the value the semaphore was constructed with. A stray release()
        call — reachable only by directly violating the AdmissionController
        contract (release() must be called exactly once, only for a
        reservation admit() returned successfully) since ChatService's real
        call sites cannot produce this — silently inflates effective capacity
        when the semaphore is not already fully free. This is not a new
        regression: the pre-Option-C _PerModelCounter had the same
        contract-reliance (a stray decrement under-counted in-flight requests,
        equally allowing one admission beyond the configured ceiling). This
        test exists so that claim is verified, not assumed."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=4),
            admission_wait_s=_FAST_WAIT_S,
        )
        # One legitimate, still-outstanding admission (never released here).
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

        failing = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        controller.release("m", failing.reserved_tokens)  # the legitimate release
        sem = controller._semaphores["m"]
        assert sem._value == 3  # 4 - 1 outstanding holder = 3 free
        controller.release("m", failing.reserved_tokens)  # contract violation: stray extra call
        assert sem._value == 4, (
            "expected the stray release to inflate _value silently (no ValueError) "
            "since 3 < bound(4) — if this now raises, BoundedSemaphore's guard "
            "changed and the docstring/report claim about its boundary is stale"
        )


class TestBoundedWaitAndCancellation:
    """Option C's core new behavior: a bounded wait instead of an instant reject,
    with exactly-once-release preserved under cancellation (design.md "Semaphore
    lifecycle analysis", paths 7-8)."""

    async def test_concurrent_admission_waits_then_succeeds_once_a_slot_frees(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=1.0,
        )
        holder = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

        waiter = asyncio.ensure_future(
            controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        )
        await asyncio.sleep(0.05)
        assert not waiter.done(), "waiter should still be blocked on the held slot"

        controller.release("m", holder.reserved_tokens)
        result = await asyncio.wait_for(waiter, timeout=1.0)
        assert result.effective_max_tokens == 10

    async def test_timeout_after_wait_elapses_raises_engine_saturated(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        # Never released -> the second admission must time out, not hang forever.
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_cancellation_while_waiting_does_not_leak_a_permit(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=5.0,
        )
        holder = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

        waiter = asyncio.ensure_future(
            controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))
        )
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        # The cancelled waiter must not have consumed (or leaked) a permit: the
        # original holder still holds the only slot, and releasing it must free
        # exactly one slot for a fresh admission — not zero (leaked) or more than
        # one (inflated, which BoundedSemaphore would raise ValueError on).
        controller.release("m", holder.reserved_tokens)
        await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

    async def test_gate2_failure_after_semaphore_acquired_releases_permit(self):
        """Lifecycle path 7: Gate 2 (context) raises after Gate 1's semaphore
        permit was already acquired. The permit must be released, not leaked —
        this is the exact hazard the original tasks.md 2.2c sketch would have
        gotten wrong (design.md "Semaphore lifecycle analysis")."""
        controller = AdmissionController(
            _registry(max_model_len=4),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
        )
        with pytest.raises(ContextTooLongError):
            await controller.admit("m", _req(), _FakeEngine(prompt_tokens=10))
        # If the permit leaked, this would time out and raise EngineSaturatedError
        # instead of succeeding immediately.
        await controller.admit("m", _req(max_tokens=1), _FakeEngine(prompt_tokens=1))

    async def test_gate3_failure_after_semaphore_acquired_releases_permit(self):
        """Lifecycle path 7, Gate 3 (KV) variant: a batch-tier request saturates
        the KV budget after Gate 1's semaphore permit was already acquired.

        max_num_seqs=2 so Gate 1 never blocks either admission by itself — the
        only way the final admission can fail is if the *failing* admission's
        Gate-1 permit was never released (a leak), which would exhaust the
        2-permit capacity and force the final admission to wait out
        admission_wait_s and raise on the sequence gate instead of succeeding.
        """
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=2),
            admission_wait_s=_FAST_WAIT_S,
        )
        # kv_safety_margin default 0.9 * capacity 30 = budget 27.
        filler_engine = _FakeEngine(prompt_tokens=5, kv_capacity_tokens=30)
        await controller.admit(
            "m", _req(max_tokens=15, priority="batch"), filler_engine
        )  # reserves 5 + 15 = 20 tokens; 7 remain in budget

        failing_engine = _FakeEngine(prompt_tokens=5, kv_capacity_tokens=30)
        with pytest.raises(EngineSaturatedError, match="KV pool"):
            # available = 27 - 20 - 5 = 2, well under the 20 requested.
            await controller.admit(
                "m", _req(max_tokens=20, priority="batch"), failing_engine
            )

        # If the failing admission's Gate-1 permit leaked, both of max_num_seqs=2's
        # permits would now be held (filler + leaked) and this would time out.
        probe_engine = _FakeEngine(prompt_tokens=5, kv_capacity_tokens=30)
        await controller.admit("m", _req(max_tokens=1, priority="batch"), probe_engine)


class TestBatchPriorityQueueing:
    """B5 (add-batch-priority-queueing): batch waits longer than interactive on
    the same per-model semaphore, and is subject to a bounded waiter cap
    interactive is not. See design.md "Cancellation and lifecycle correctness"
    for the path numbering referenced below.
    """

    async def test_batch_waits_then_succeeds_once_a_slot_frees(self):
        """Batch uses batch_admission_wait_s, not admission_wait_s, and is
        admitted once the holder releases — same mechanism as interactive's
        TestBoundedWaitAndCancellation case, batch-priority variant."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=0.01,  # deliberately too short for interactive to survive
            batch_admission_wait_s=1.0,
        )
        holder = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

        waiter = asyncio.ensure_future(
            controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        assert not waiter.done(), "batch waiter should still be blocked on the held slot"

        controller.release("m", holder.reserved_tokens)
        result = await asyncio.wait_for(waiter, timeout=1.0)
        assert result.effective_max_tokens == 10

    async def test_batch_timeout_raises_engine_saturated_with_retry_after(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        # Never released -> the second batch admission must time out.
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit") as exc_info:
            await controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        assert exc_info.value.retry_after_s == 1.0

    async def test_interactive_keeps_its_own_short_wait_bound_unaffected_by_batch_default(self):
        """Interactive must not be slowed down by batch_admission_wait_s existing
        at all — it keeps using admission_wait_s exactly as B4 shipped it."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=_FAST_WAIT_S,
            batch_admission_wait_s=30.0,  # would hang the test if interactive used this
        )
        await controller.admit("m", _req(max_tokens=10, priority="interactive"), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError, match="concurrent-sequence limit"):
            await controller.admit(
                "m", _req(max_tokens=10, priority="interactive"), _FakeEngine(prompt_tokens=1)
            )

    async def test_batch_waiter_cap_overflow_rejects_immediately_without_acquiring_semaphore(self):
        """A batch request arriving when the waiter cap is already full must be
        rejected before ever touching the semaphore — not even attempt a wait."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=5.0,  # long enough that a real wait would make this test hang
            batch_waiter_multiplier=1,  # cap = 1 * max_num_seqs(1) = 1
        )
        # Hold the only permit so subsequent batch admissions must queue.
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))

        # First waiter fills the cap (cap=1): it starts waiting on the semaphore.
        first_waiter = asyncio.ensure_future(
            controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        assert not first_waiter.done()

        # Second waiter arrives while the cap is already full -> immediate 429,
        # not a 5-second wait. If this test hangs for ~5s, the cap check is
        # missing or is running after sem.acquire() instead of before it.
        with pytest.raises(EngineSaturatedError, match="batch queue is full"):
            await asyncio.wait_for(
                controller.admit(
                    "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
                ),
                timeout=1.0,
            )

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

    async def test_batch_waiter_count_decremented_after_successful_admission(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=5),
            batch_admission_wait_s=1.0,
        )
        for _ in range(3):
            await controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        assert controller._batch_waiters.current("m") == 0

    async def test_batch_waiter_count_decremented_after_timeout(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=_FAST_WAIT_S,
        )
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        with pytest.raises(EngineSaturatedError):
            await controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        assert controller._batch_waiters.current("m") == 0

    async def test_batch_waiter_count_decremented_after_cancellation(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=5.0,
        )
        holder = await controller.admit(
            "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
        )

        waiter = asyncio.ensure_future(
            controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        assert controller._batch_waiters.current("m") == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert controller._batch_waiters.current("m") == 0
        # And no semaphore permit leaked either: releasing the original holder
        # must free exactly one slot for a fresh admission.
        controller.release("m", holder.reserved_tokens)
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))

    async def test_no_waiter_count_leak_after_gate2_rejection(self):
        """A batch request that acquires its semaphore permit then fails Gate 2
        (context) must not leave a stale batch-waiter count behind — the
        counter is decremented at the wait site itself, before Gate 2/3 ever
        run, so this documents that ordering holds under a real Gate 2 failure."""
        controller = AdmissionController(
            _registry(max_model_len=4),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=1.0,
        )
        with pytest.raises(ContextTooLongError):
            await controller.admit(
                "m", _req(priority="batch"), _FakeEngine(prompt_tokens=10)
            )
        assert controller._batch_waiters.current("m") == 0
        # And the semaphore permit wasn't leaked either (existing B4 invariant).
        await controller.admit(
            "m", _req(max_tokens=1, priority="batch"), _FakeEngine(prompt_tokens=1)
        )

    async def test_no_waiter_count_leak_after_gate3_rejection(self):
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=2),
            batch_admission_wait_s=1.0,
        )
        filler_engine = _FakeEngine(prompt_tokens=5, kv_capacity_tokens=30)
        await controller.admit("m", _req(max_tokens=15, priority="batch"), filler_engine)

        failing_engine = _FakeEngine(prompt_tokens=5, kv_capacity_tokens=30)
        with pytest.raises(EngineSaturatedError, match="KV pool"):
            await controller.admit(
                "m", _req(max_tokens=20, priority="batch"), failing_engine
            )
        assert controller._batch_waiters.current("m") == 0

    async def test_multiple_models_have_independent_batch_waiter_state(self):
        registry = ModelRegistry(
            [ModelEntry(name="a", model_path="test/a"), ModelEntry(name="b", model_path="test/b")]
        )
        controller = AdmissionController(
            registry,
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=5.0,
        )
        await controller.admit("a", _req(model="a", max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))
        await controller.admit("b", _req(model="b", max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))

        waiter_a = asyncio.ensure_future(
            controller.admit(
                "a", _req(model="a", max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        assert controller._batch_waiters.current("a") == 1
        assert controller._batch_waiters.current("b") == 0

        waiter_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_a

    async def test_fifo_order_preserved_not_priority_preemption(self):
        """Option 1 is FIFO on the shared semaphore, not a priority-preemptive
        queue: batch requests already waiting are not skipped over by a later-
        arriving interactive request. This documents the approved tradeoff
        rather than inventing preemption."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            admission_wait_s=2.0,
            batch_admission_wait_s=2.0,
        )
        holder = await controller.admit("m", _req(max_tokens=10), _FakeEngine(prompt_tokens=1))

        batch_waiter = asyncio.ensure_future(
            controller.admit(
                "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        interactive_waiter = asyncio.ensure_future(
            controller.admit(
                "m", _req(max_tokens=10, priority="interactive"), _FakeEngine(prompt_tokens=1)
            )
        )
        await asyncio.sleep(0.05)
        assert not batch_waiter.done() and not interactive_waiter.done()

        # Free exactly one slot: CPython's asyncio.Semaphore wakes waiters in
        # FIFO (arrival) order, so the batch waiter -- which arrived first --
        # must be the one admitted, not the later-arriving interactive one.
        controller.release("m", holder.reserved_tokens)
        done, pending = await asyncio.wait(
            {batch_waiter, interactive_waiter}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
        )
        assert batch_waiter in done
        assert interactive_waiter in pending

        # Clean up the still-pending interactive waiter and the batch holder.
        result = await batch_waiter
        controller.release("m", result.reserved_tokens)
        interactive_result = await interactive_waiter
        controller.release("m", interactive_result.reserved_tokens)

    async def test_batch_waiter_cap_boundary_exact_capacity_admits_the_last_one(self):
        """cap = multiplier * max_num_seqs = 2 * 1 = 2: exactly 2 concurrent
        batch waiters must be accepted (queued), a 3rd must be rejected
        immediately -- verifying the '>=' boundary, not an off-by-one '>'."""
        controller = AdmissionController(
            _registry(),
            tier=_FakeTier(max_model_len_cap=4096, max_num_seqs=1),
            batch_admission_wait_s=5.0,
            batch_waiter_multiplier=2,
        )
        await controller.admit("m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1))

        waiters = [
            asyncio.ensure_future(
                controller.admit(
                    "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
                )
            )
            for _ in range(2)
        ]
        await asyncio.sleep(0.05)
        assert controller._batch_waiters.current("m") == 2
        assert not any(w.done() for w in waiters)

        with pytest.raises(EngineSaturatedError, match="batch queue is full"):
            await asyncio.wait_for(
                controller.admit(
                    "m", _req(max_tokens=10, priority="batch"), _FakeEngine(prompt_tokens=1)
                ),
                timeout=0.5,
            )

        for w in waiters:
            w.cancel()
        for w in waiters:
            with pytest.raises(asyncio.CancelledError):
                await w

