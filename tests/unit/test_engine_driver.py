"""Unit tests for EngineDriver (see engines/driver.py, DEC-038/DEC-039).

Uses a fake llm_engine that behaves like vLLM's synchronous engine closely
enough to exercise the driver's demultiplexing logic: add_request registers a
pre-scripted sequence of RequestOutputs (keyed by prompt) and step() advances
every active request by one scripted output per call, mirroring how vLLM
interleaves output across concurrently in-flight requests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from inference_x.engines.driver import EngineDriver, EngineDriverDeadError


@dataclass
class _FakeCompletion:
    text: str = ""
    finish_reason: str | None = None
    token_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class _FakeRequestOutput:
    request_id: str
    outputs: list[_FakeCompletion]
    finished: bool
    prompt_token_ids: list[int] = field(default_factory=lambda: [1, 2, 3])


class FakeLLMEngine:
    """Fake vLLM sync engine: scripts are registered by prompt, not request_id,
    since only the driver ever generates request_ids."""

    def __init__(self) -> None:
        self._scripts: dict[str, list[tuple[str, bool]]] = {}
        self._active: dict[str, list[tuple[str, bool]]] = {}
        self.added: list[tuple[str, str]] = []
        self.step_calls = 0
        self.step_raises: Exception | None = None
        # Set to gate step_raises: step() returns an empty (no-op) result instead
        # of raising until at least this many add_request calls have landed. Lets
        # a test guarantee N requests are registered with the fake engine before
        # the failure fires, without any sleep-based timing -- see
        # test_step_exception_is_broadcast_to_pending_completion_futures.
        self.fail_after_registered: int | None = None

    def register_script(self, prompt: str, steps: list[tuple[str, bool]]) -> None:
        """steps: list of (cumulative_text, finished) tuples, one per step() call."""
        self._scripts[prompt] = list(steps)

    def add_request(self, request_id: str, prompt: str, sampling) -> None:
        self.added.append((request_id, prompt))
        self._active[request_id] = self._scripts.pop(prompt)

    def step(self) -> list[_FakeRequestOutput]:
        self.step_calls += 1
        if self.step_raises is not None:
            if self.fail_after_registered is not None and len(self.added) < self.fail_after_registered:
                return []
            raise self.step_raises
        outputs = []
        for request_id in list(self._active):
            remaining = self._active[request_id]
            if not remaining:
                continue
            text, finished = remaining.pop(0)
            outputs.append(
                _FakeRequestOutput(
                    request_id=request_id,
                    # One token id per character: deliberately different from the
                    # whitespace word count DEC-049 removed, so a test asserting
                    # engine-accounted usage cannot pass by accident.
                    outputs=[_FakeCompletion(text=text, token_ids=list(range(len(text))))],
                    finished=finished,
                )
            )
            if finished:
                del self._active[request_id]
        return outputs


def _drain_stream(out_queue, timeout: float = 5.0) -> list:
    chunks = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            chunk = out_queue.get(timeout=0.5)
        except Exception:
            continue
        if chunk is None:
            return chunks
        chunks.append(chunk)
    raise AssertionError("stream did not terminate within timeout")


def _drain_text(out_queue, timeout: float = 5.0) -> list[str]:
    """Content deltas only, dropping the terminal metadata chunk (DEC-049)."""
    return [c.content for c in _drain_stream(out_queue, timeout) if c.content]


def test_single_completion_request():
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("Hello", False), ("Hello world", True)])
    driver = EngineDriver(fake, threading.Lock())
    try:
        future = driver.submit_complete("prompt-A", object())
        output = future.result(timeout=5.0)
        assert output.outputs[0].text == "Hello world"
        assert output.finished is True
    finally:
        driver.shutdown()


def test_two_concurrent_completions_both_return_correct_distinct_output():
    """Regression test for the DEC-038 race: 2 concurrent non-streaming
    requests must both complete with their own content, never dropped or
    swapped."""
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("A-partial", False), ("A-final", True)])
    fake.register_script("prompt-B", [("B-partial", False), ("B-final", True)])
    driver = EngineDriver(fake, threading.Lock())
    try:
        future_a = driver.submit_complete("prompt-A", object())
        future_b = driver.submit_complete("prompt-B", object())

        output_a = future_a.result(timeout=5.0)
        output_b = future_b.result(timeout=5.0)

        assert output_a.outputs[0].text == "A-final"
        assert output_b.outputs[0].text == "B-final"
    finally:
        driver.shutdown()


def test_stream_yields_incremental_deltas():
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("Hello", False), ("Hello world", True)])
    driver = EngineDriver(fake, threading.Lock())
    try:
        out_queue = driver.submit_stream("prompt-A", object())
        chunks = _drain_stream(out_queue)

        # Content deltas are unchanged; the driver now appends one terminal
        # chunk carrying engine-accounted metadata before the None sentinel.
        assert [c.content for c in chunks if c.content] == ["Hello", " world"]

        terminal = chunks[-1]
        assert terminal.content == ""
        assert terminal.finish_reason == "stop"
        assert terminal.usage is not None
        # Engine-accounted: one id per character of "Hello world" (11), not the
        # whitespace word count (2) this replaced.
        assert terminal.usage.completion_tokens == 11
        assert terminal.usage.prompt_tokens == 3
        assert terminal.usage.total_tokens == 14
    finally:
        driver.shutdown()


def test_two_concurrent_streams_do_not_cross_contaminate():
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("Hi", False), ("Hi there", True)])
    fake.register_script("prompt-B", [("Yo", False), ("Yo dude", True)])
    driver = EngineDriver(fake, threading.Lock())
    try:
        queue_a = driver.submit_stream("prompt-A", object())
        queue_b = driver.submit_stream("prompt-B", object())

        chunks_a = _drain_text(queue_a)
        chunks_b = _drain_text(queue_b)

        assert "".join(chunks_a) == "Hi there"
        assert "".join(chunks_b) == "Yo dude"
    finally:
        driver.shutdown()


def test_step_exception_is_broadcast_to_pending_completion_futures():
    """Both requests must be registered with the fake engine before step() is
    allowed to raise (fake.fail_after_registered), so this no longer depends on
    scheduling luck to land both submissions ahead of the failure -- see DEC-043
    for the pre-fix flake this replaces (was previously ~40% flaky: a second
    submission could lose the race against the driver thread's failure/exit and
    hang until timeout instead of landing in the broadcast)."""
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("partial", False), ("final", True)])
    fake.register_script("prompt-B", [("partial", False), ("final", True)])
    fake.step_raises = RuntimeError("simulated CUDA failure")
    fake.fail_after_registered = 2
    driver = EngineDriver(fake, threading.Lock())
    try:
        future_a = driver.submit_complete("prompt-A", object())
        future_b = driver.submit_complete("prompt-B", object())

        with pytest.raises(RuntimeError, match="simulated CUDA failure"):
            future_a.result(timeout=5.0)
        with pytest.raises(RuntimeError, match="simulated CUDA failure"):
            future_b.result(timeout=5.0)

        # `_broadcast_exception` sets `_dead` before it resolves any pending
        # future/queue, so this is already guaranteed true here -- no polling needed.
        assert driver.is_dead is True
    finally:
        driver.shutdown()


def test_submit_after_driver_death_raises_immediately():
    """DEC-043 regression: once the driver is dead, submit_complete rejects
    immediately with EngineDriverDeadError instead of silently enqueuing a
    request that no thread is left to serve, which previously hung the caller
    until its completion/stream timeout."""
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("partial", False), ("final", True)])
    fake.step_raises = RuntimeError("simulated failure")
    driver = EngineDriver(fake, threading.Lock())
    try:
        future_a = driver.submit_complete("prompt-A", object())
        with pytest.raises(RuntimeError, match="simulated failure"):
            future_a.result(timeout=5.0)
        assert driver.is_dead is True

        start = time.monotonic()
        with pytest.raises(EngineDriverDeadError):
            driver.submit_complete("prompt-late", object())
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, "submit after death must reject immediately, not time out"

        with pytest.raises(EngineDriverDeadError):
            driver.submit_stream("prompt-late-stream", object())
    finally:
        driver.shutdown()


def test_step_exception_is_broadcast_to_pending_stream():
    fake = FakeLLMEngine()
    fake.register_script("prompt-A", [("partial", False), ("final", True)])
    fake.step_raises = RuntimeError("simulated failure")
    driver = EngineDriver(fake, threading.Lock())
    try:
        out_queue = driver.submit_stream("prompt-A", object())
        item = out_queue.get(timeout=5.0)
        assert isinstance(item, BaseException)
        assert "simulated failure" in str(item)
    finally:
        driver.shutdown()


def test_shutdown_stops_the_driver_thread():
    fake = FakeLLMEngine()
    driver = EngineDriver(fake, threading.Lock())
    driver.shutdown(timeout=5.0)
    assert not driver._thread.is_alive()
