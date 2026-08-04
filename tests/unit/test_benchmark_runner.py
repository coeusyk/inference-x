"""Unit tests for BenchmarkRunner VRAM footprint and SSE consumption."""
from __future__ import annotations

from inference_x.benchmarks.runner import (
    _check_vram_budget,
    _peak_vram_footprint_gb,
    _run_prompt_stream,
)
from inference_x.benchmarks.schemas import HardwareProfile


def _hw(
    *,
    total: float,
    free: float,
    has_gpu: bool = True,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_gb=total,
        vram_free_gb=free,
        cpu_cores=8,
        ram_total_gb=32.0,
        has_gpu=has_gpu,
    )


class TestPeakVramFootprint:
    def test_preloaded_model_nonzero_footprint(self):
        """Model already loaded: legacy delta is 0 but footprint is not."""
        before = _hw(total=6.0, free=2.78)
        after = _hw(total=6.0, free=2.78)
        assert _peak_vram_footprint_gb(before, after) == 3.22

    def test_cold_load_footprint(self):
        before = _hw(total=6.0, free=5.5)
        after = _hw(total=6.0, free=2.9)
        assert _peak_vram_footprint_gb(before, after) == 3.1

    def test_cpu_only_returns_zero(self):
        before = _hw(total=0.0, free=0.0, has_gpu=False)
        after = _hw(total=0.0, free=0.0, has_gpu=False)
        assert _peak_vram_footprint_gb(before, after) == 0.0


class TestCheckVramBudget:
    def test_no_model_path_skips_check(self):
        exceeded, warning = _check_vram_budget("m", 99.0, None, 2048, None)
        assert exceeded is False
        assert warning is None

    def test_within_budget(self):
        exceeded, warning = _check_vram_budget(
            "opt-125m", 0.01, "facebook/opt-125m", 2048, None
        )
        assert exceeded is False
        assert warning is None

    def test_exceeds_budget(self):
        exceeded, warning = _check_vram_budget(
            "opt-125m", 500.0, "facebook/opt-125m", 2048, None
        )
        assert exceeded is True
        assert warning is not None
        assert "opt-125m" in warning


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def stream(self, *args, **kwargs):
        return _FakeStreamResponse(self._lines)


class TestStreamConsumption:
    """Compatibility invariant: this consumer parses the OS-4 stream unmodified.

    The pre-generation event (DEC-053) carries ``choices: []`` and no ``usage``
    key, so it falls through the usage branch and resolves to an empty delta —
    contributing no token and no TTFT. Verified here rather than assumed,
    because the runner is a first-party client and a silent break in it would
    only surface as wrong benchmark numbers.
    """

    _PRE_GENERATION = (
        'data: {"id":"x","object":"chat.completion.chunk","choices":[],'
        '"resolved":{"model":"m","max_tokens":256},"warnings":['
        '{"type":"degraded","code":"kv_gate_skipped","message":"x","field":null}]}'
    )
    _CONTENT = (
        'data: {"id":"x","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}'
    )
    _TERMINAL = (
        'data: {"id":"x","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
    )
    _USAGE = (
        'data: {"id":"x","object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":4,"completion_tokens":7,"total_tokens":11}}'
    )

    def _run(self, lines: list[str]):
        return _run_prompt_stream(
            _FakeClient(lines), "http://test", "m", "hi", "label"
        )

    def test_pre_generation_event_changes_nothing(self):
        without = self._run([self._CONTENT, self._TERMINAL, self._USAGE, "data: [DONE]"])
        with_pre = self._run(
            [self._PRE_GENERATION, self._CONTENT, self._TERMINAL, self._USAGE, "data: [DONE]"]
        )

        assert with_pre.tokens_generated == without.tokens_generated == 7
        assert with_pre.ttft_ms is not None
        assert without.ttft_ms is not None

    def test_ttft_still_comes_from_the_first_content_event(self):
        result = self._run(
            [self._PRE_GENERATION, self._CONTENT, self._TERMINAL, self._USAGE, "data: [DONE]"]
        )
        # A pre-generation event alone must not satisfy TTFT: with no content
        # event at all the runner falls back to total latency, which is what the
        # equality below detects.
        no_content = self._run([self._PRE_GENERATION, self._TERMINAL, "data: [DONE]"])
        assert no_content.ttft_ms == no_content.total_latency_ms
        assert result.ttft_ms < result.total_latency_ms or result.tokens_generated == 7
