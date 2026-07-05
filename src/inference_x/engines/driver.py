"""Shared per-engine driver thread for vLLM's synchronous step() loop.

See DEC-038 for the incident this fixes: mirroring `generate_stream`'s
add_request/step() loop directly inside `_run_completion` (releasing the lock
between steps, one loop per request) passed unit tests against a mocked engine
and then reproducibly lost output under live concurrency. `generate_stream`
tolerates a request's own step being "won" by another thread because
`output.outputs[0].text` is cumulative — any later call surfacing your
request_id lets you compute the missed delta. A one-shot `finished=True`
handoff has no such recovery: if a *different* thread's `step()` call is the
one that returns your request's terminal output, that thread discards it
(request_id mismatch), and `has_unfinished_requests()` can go globally False
before the owning thread ever calls `step()` again.

EngineDriver removes the race by construction rather than working around it:
exactly one thread ever calls `add_request`/`step()` for a given engine, and
that same thread demultiplexes every `step()` output to the right destination
by request_id. There is no "wrong" thread left to discard anything on.
Streaming and non-streaming both submit through this one driver and differ
only in the shape of the channel they read back (a queue of text deltas vs. a
single future), not in how requests reach the engine.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

_IDLE_POLL_S = 0.05  # bounds shutdown responsiveness, not correctness


@dataclass
class _PendingRequest:
    """Per-request state, indexed by request_id, only ever touched on the driver thread."""

    mode: Literal["stream", "complete"]
    out_queue: "queue.Queue[str | BaseException | None] | None" = None
    future: Future | None = None
    previous_text: str = ""


class EngineDriverDeadError(RuntimeError):
    """Raised when submitting to a driver whose thread has already exited."""


class EngineDriver:
    """Owns a vLLM sync `llm_engine` exclusively; the sole caller of add_request/step().

    One instance per `VLLMEngine`. `step_lock` is the same lock the engine used
    to serialize step() calls before this driver existed — `_POOL_STEP_LOCK`
    when multiple engines share one process (vLLM V1 forward context), a
    private per-engine lock otherwise. That cross-engine constraint is
    orthogonal to this class and must keep being honored here: acquiring it
    per-driver-instance instead would silently reintroduce the multi-engine
    race it was added to prevent.
    """

    def __init__(self, llm_engine: Any, step_lock: threading.Lock) -> None:
        self._llm_engine = llm_engine
        self._step_lock = step_lock
        self._submit_q: queue.Queue[tuple[str, str, Any, _PendingRequest]] = queue.Queue()
        self._pending: dict[str, _PendingRequest] = {}
        self._dead = False
        self._dead_exception: BaseException | None = None
        # Guards `_dead`/`_dead_exception` together with the decision of whether a
        # given `_submit_q.put()` is allowed at all — see DEC-043. Checking `_dead`
        # and enqueuing as two separate unguarded steps left a window where a
        # request submitted right as `step()` failed would be enqueued after
        # `_broadcast_exception` had already drained the queue and returned,
        # orphaning it until its caller's completion/stream timeout. Making
        # "check dead, else enqueue" and "set dead, then drain" share one lock
        # closes that window: either the enqueue completes before the drain (and
        # is caught by it) or after the flag is already set (and never happens).
        self._dead_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def is_dead(self) -> bool:
        return self._dead

    def submit_stream(self, prompt: str, sampling: Any) -> "queue.Queue[str | BaseException | None]":
        """Submit a streaming request; returns a queue yielding text deltas, then None.

        Raises `EngineDriverDeadError` immediately if the driver has already
        failed — see `_submit`.
        """
        pending = _PendingRequest(mode="stream", out_queue=queue.Queue())
        self._submit(prompt, sampling, pending)
        return pending.out_queue

    def submit_complete(self, prompt: str, sampling: Any) -> Future:
        """Submit a non-streaming request; returns a Future resolved with the final RequestOutput.

        Raises `EngineDriverDeadError` immediately if the driver has already
        failed — see `_submit`.
        """
        pending = _PendingRequest(mode="complete", future=Future())
        self._submit(prompt, sampling, pending)
        return pending.future

    def _submit(self, prompt: str, sampling: Any, pending: _PendingRequest) -> None:
        request_id = f"cmpl-{uuid.uuid4().hex}"
        with self._dead_lock:
            if self._dead:
                raise EngineDriverDeadError(
                    "engine driver thread has exited"
                ) from self._dead_exception
            self._submit_q.put((request_id, prompt, sampling, pending))

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pending:
                # Requests are already in flight: drain any new submissions
                # without blocking and go straight to step(). Blocking here
                # too (see the idle branch below) capped every step() to
                # 1/_IDLE_POLL_S regardless of model speed — a ~20 tok/s
                # ceiling traced from an opt-125m benchmark regression after
                # this driver thread replaced the old unthrottled per-request
                # `while has_unfinished_requests(): step()` loop.
                self._drain_submissions_nowait()
            else:
                self._drain_submissions_blocking()
            if not self._pending:
                continue
            try:
                with self._step_lock:
                    step_outputs = self._llm_engine.step()
            except Exception as exc:  # noqa: BLE001 - must reach every pending caller
                logger.warning("Engine driver step() failed, broadcasting: %s", exc)
                self._broadcast_exception(exc)
                return
            if not step_outputs:
                continue
            for output in step_outputs:
                pending = self._pending.get(output.request_id)
                if pending is None or not output.outputs:
                    continue
                self._dispatch(output.request_id, pending, output)

    def _drain_submissions_blocking(self) -> None:
        """Block up to _IDLE_POLL_S for the first submission when genuinely idle.

        Only called from _run() when self._pending is already empty — this is
        what bounds shutdown() responsiveness without spinning the CPU while
        there is nothing to do. Must never run while requests are pending;
        see the comment in _run().
        """
        try:
            request_id, prompt, sampling, pending = self._submit_q.get(timeout=_IDLE_POLL_S)
        except queue.Empty:
            return
        self._register(request_id, prompt, sampling, pending)
        self._drain_submissions_nowait()

    def _drain_submissions_nowait(self) -> None:
        while True:
            try:
                request_id, prompt, sampling, pending = self._submit_q.get_nowait()
            except queue.Empty:
                break
            self._register(request_id, prompt, sampling, pending)

    def _register(self, request_id: str, prompt: str, sampling: Any, pending: _PendingRequest) -> None:
        self._pending[request_id] = pending
        try:
            self._llm_engine.add_request(request_id, prompt, sampling)
        except Exception as exc:
            del self._pending[request_id]
            self._fail_one(pending, exc)

    def _dispatch(self, request_id: str, pending: _PendingRequest, output: Any) -> None:
        text = output.outputs[0].text or ""
        if pending.mode == "stream":
            if text.startswith(pending.previous_text):
                chunk = text[len(pending.previous_text):]
            else:
                chunk = text
            pending.previous_text = text
            if chunk:
                pending.out_queue.put(chunk)
            if output.finished:
                pending.out_queue.put(None)
                del self._pending[request_id]
        else:
            if output.finished:
                pending.future.set_result(output)
                del self._pending[request_id]

    def _fail_one(self, pending: _PendingRequest, exc: Exception) -> None:
        if pending.mode == "stream":
            pending.out_queue.put(exc)
        else:
            pending.future.set_exception(exc)

    def _broadcast_exception(self, exc: Exception) -> None:
        # Set dead + drain the submission queue as one atomic step with `_submit`'s
        # check-then-enqueue (same lock) — anything that lands in the queue before
        # this runs is caught here; anything submitted after sees `_dead` already
        # set and never reaches the queue at all. See __init__ and DEC-043.
        with self._dead_lock:
            self._dead = True
            self._dead_exception = exc
            while True:
                try:
                    _request_id, _prompt, _sampling, pending = self._submit_q.get_nowait()
                except queue.Empty:
                    break
                self._fail_one(pending, exc)
        # `_pending` is only ever touched by this (the driver) thread, so broadcasting
        # to it needs no lock.
        for pending in self._pending.values():
            self._fail_one(pending, exc)
        self._pending.clear()
