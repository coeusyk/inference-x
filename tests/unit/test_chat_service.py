"""Unit tests for ChatService (Phase 4: multi-model engine pool)."""
import asyncio
import json

import pytest

from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.admission import AdmissionController, AdmissionResult
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatStreamChunk,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*names: str) -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=n, model_path=f"test/{n}") for n in names])


def _make_pool(*names: str, healthy: bool = True, raise_on_generate: bool = False) -> EnginePool:
    return EnginePool({n: _StubEngine(healthy=healthy, raise_on_generate=raise_on_generate) for n in names})


def _make_service(
    model_name: str = "test",
    healthy: bool = True,
    raise_on_generate: bool = False,
    engine: BaseEngine | None = None,
) -> ChatService:
    registry = _make_registry(model_name)
    router = TaskRouter(registry, model_name)
    if engine is not None:
        pool = EnginePool({model_name: engine})
    else:
        pool = _make_pool(model_name, healthy=healthy, raise_on_generate=raise_on_generate)
    return ChatService(engine_pool=pool, registry=registry, router=router)


class _CountingAdmission:
    """Wraps a real AdmissionController, recording admit()/release() calls.

    Delegates admit()/release() to the wrapped controller unchanged, so the
    real KV/sequence trackers update exactly as they would in production —
    this only adds observable records of what admit() returned and what
    release() received, to make the "exactly one release()", "never released
    twice", and "release() receives the same reservation admit() produced"
    invariants directly assertable (SEV-A1 fix, fix-admission-reservation-leak).
    """

    def __init__(self, inner: AdmissionController) -> None:
        self.inner = inner
        self.admit_results: list[AdmissionResult] = []
        self.release_calls: list[tuple[str, int]] = []

    def admit(self, *args, **kwargs):
        result = self.inner.admit(*args, **kwargs)
        self.admit_results.append(result)
        return result

    def release(self, routed_model: str, reserved_tokens: int) -> None:
        self.release_calls.append((routed_model, reserved_tokens))
        self.inner.release(routed_model, reserved_tokens)


class _StubEngine(BaseEngine):
    def __init__(self, healthy: bool = True, raise_on_generate: bool = False) -> None:
        self._healthy = healthy
        self._raise = raise_on_generate

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self._raise:
            raise RuntimeError("stub generation error")
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def generate_stream(self, request: ChatCompletionRequest):
        if self._raise:
            raise RuntimeError("stub streaming error")
        yield ChatStreamChunk(content="ok")
        yield ChatStreamChunk(
            content="",
            finish_reason="stop",
            usage=ChatCompletionUsage(
                prompt_tokens=3, completion_tokens=1, total_tokens=4
            ),
        )

    def is_healthy(self) -> bool:
        return self._healthy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatService:
    def _req(self, model: str = "test") -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content="hello")],
        )

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        svc = _make_service()
        resp = await svc.complete(self._req())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_complete_attaches_resolved_and_warnings(self):
        """The service attaches them; the engine never constructs them (DEC-047)."""
        svc = _make_service()
        resp = await svc.complete(self._req())
        assert resp.resolved is not None
        assert resp.resolved.model == "test"
        assert resp.resolved.max_tokens == 512
        # The stub engine builds its response without either field.
        assert isinstance(resp.warnings, list)

    @pytest.mark.asyncio
    async def test_complete_resolved_reports_the_clamped_value(self):
        svc = _make_service()
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=4000,
            max_context_tokens=200,
        )
        resp = await svc.complete(req)
        assert resp.resolved is not None
        assert resp.resolved.max_tokens < 4000
        assert {w.code for w in resp.warnings} == {"max_tokens_clamped_to_context"} | {
            w.code for w in resp.warnings if w.type == "degraded"
        }

    @pytest.mark.asyncio
    async def test_strict_does_not_change_an_accepted_execution(self):
        """DEC-052 §3: accepted under both modes -> byte-identical output.

        This is the CI-checkable form of "strict may only convert a substitution
        into a rejection". The engine echoes the parameters it was handed, so if
        `strict` ever reaches execution — directly, or by changing what admission
        hands over — the bytes diverge and this fails. An engine returning a
        constant would make the assertion unfalsifiable.
        """

        class _EchoEngine(BaseEngine):
            async def generate(self, request: ChatCompletionRequest):
                echoed = (
                    f"max_tokens={request.max_tokens} seed={request.seed} "
                    f"temperature={request.temperature} top_p={request.top_p} "
                    f"priority={request.priority} strict={request.strict}"
                )
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatCompletionMessage(content=echoed),
                            finish_reason="stop",
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=1, completion_tokens=1, total_tokens=2
                    ),
                )

            async def generate_stream(self, request):  # pragma: no cover - unused
                raise AssertionError("not used")
                yield

            def is_healthy(self) -> bool:
                return True

        svc = _make_service(engine=_EchoEngine())

        def _req(strict: bool) -> ChatCompletionRequest:
            return ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="hello")],
                seed=42,
                strict=strict,
            )

        lenient = await svc.complete(_req(strict=False))
        strict = await svc.complete(_req(strict=True))

        # `strict` is the one field expected to differ — it is the request flag
        # itself, not something the flag changed about execution.
        assert lenient.choices[0].message.content.encode() == strict.choices[
            0
        ].message.content.replace("strict=True", "strict=False").encode()
        assert lenient.resolved == strict.resolved

    @pytest.mark.asyncio
    async def test_the_echo_guard_can_actually_fail(self):
        """Guards the test above: prove the echo detects a parameter difference.

        Without this, a later refactor could make the echo constant again and the
        DEC-052 §3 assertion would keep passing while checking nothing.
        """
        svc = _make_service()
        a = await svc.complete(
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=100,
            )
        )
        b = await svc.complete(
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=200,
            )
        )
        assert a.resolved != b.resolved

    @pytest.mark.asyncio
    async def test_stream_response_formats_sse(self):
        svc = _make_service()
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )
        events = [event async for event in svc.stream_response(req)]
        assert events[-1] == "data: [DONE]\n\n"
        # events[0] is the pre-generation metadata event (DEC-053); the first
        # content event follows it.
        payload = json.loads(events[1].removeprefix("data: ").strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_complete_propagates_runtime_error(self):
        svc = _make_service(raise_on_generate=True)
        with pytest.raises(RuntimeError, match="stub generation error"):
            await svc.complete(self._req())

    def test_engine_healthy_delegates_to_pool(self):
        assert _make_service(healthy=True).engine_healthy() is True
        assert _make_service(healthy=False).engine_healthy() is False

    def test_loaded_models_returns_pool_contents(self):
        registry = _make_registry("alpha", "beta")
        router = TaskRouter(registry, "alpha")
        pool = _make_pool("alpha", "beta")
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        assert svc.loaded_models() == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_unregistered_model_falls_back_to_default(self):
        """When a client requests an unknown model, routing falls back to default."""
        svc = _make_service(model_name="test")
        req = self._req(model="unknown-model")
        resp = await svc.complete(req)
        assert resp is not None

    @pytest.mark.asyncio
    async def test_model_not_in_pool_raises_value_error(self):
        """Requesting a registered but unloaded model raises ValueError."""
        registry = _make_registry("model-a", "model-b")
        router = TaskRouter(registry, "model-a")
        # Pool only has model-a loaded
        pool = EnginePool({"model-a": _StubEngine()})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)
        req = self._req(model="model-b")
        with pytest.raises(ValueError, match="Routed to model"):
            await svc.complete(req)

    @pytest.mark.asyncio
    async def test_multi_model_pool_dispatches_correctly(self):
        """Each model in the pool gets its own engine calls."""
        registry = _make_registry("alpha", "beta")
        router = TaskRouter(registry, "alpha")

        class _NamedEngine(BaseEngine):
            def __init__(self, name: str) -> None:
                self._name = name

            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                return ChatCompletionResponse(
                    model=self._name,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(content=f"from {self._name}"),
                        finish_reason="stop",
                    )],
                    usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            async def generate_stream(self, request: ChatCompletionRequest):
                yield ChatStreamChunk(content=f"from {self._name}")
                yield ChatStreamChunk(content="", finish_reason="stop")

            def is_healthy(self) -> bool:
                return True

        pool = EnginePool({"alpha": _NamedEngine("alpha"), "beta": _NamedEngine("beta")})
        svc = ChatService(engine_pool=pool, registry=registry, router=router)

        resp_a = await svc.complete(self._req(model="alpha"))
        resp_b = await svc.complete(self._req(model="beta"))

        assert resp_a.choices[0].message.content == "from alpha"
        assert resp_b.choices[0].message.content == "from beta"


def _parse(events: list[str]) -> list[object]:
    """Decode an SSE event list to payloads, keeping ``[DONE]`` as a marker."""
    out: list[object] = []
    for event in events:
        assert event.startswith("data: "), f"malformed SSE event: {event!r}"
        assert event.endswith("\n\n"), f"event not terminated by blank line: {event!r}"
        body = event.removeprefix("data: ").strip()
        out.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return out


class TestStreamingContract:
    """Authoritative protocol specification for the SSE event sequence.

    DEC-049 fixed the order; DEC-053 extended it at the head with exactly one
    pre-generation metadata event and forbade anything after the usage event.

    These assertions are the contract. If an implementation change makes one
    fail, the implementation is wrong — do not relax the assertion to match it.
    Each test asserts the COMPLETE ordered sequence and the exact event count,
    not merely that some expected substring appears somewhere.
    """

    def _req(self, **kw) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
            **kw,
        )

    @pytest.mark.asyncio
    async def test_default_stream_has_no_usage_event(self):
        """No stream_options -> resolution, content, terminal, [DONE]. Exactly four."""
        events = _parse(
            [e async for e in _make_service().stream_response(self._req())]
        )

        assert len(events) == 4

        resolution, content, terminal, done = events

        # Pre-generation event: empty choices, resolved block, warnings array.
        assert resolution["object"] == "chat.completion.chunk"
        assert resolution["choices"] == []
        assert resolution["resolved"]["model"] == "test"
        assert "warnings" in resolution
        assert "usage" not in resolution

        assert content["object"] == "chat.completion.chunk"
        assert content["choices"][0]["delta"] == {"content": "ok"}
        assert content["choices"][0]["finish_reason"] is None
        assert "usage" not in content

        assert terminal["choices"][0]["delta"] == {}
        assert terminal["choices"][0]["finish_reason"] == "stop"
        assert "usage" not in terminal

        assert done == "[DONE]"

        # All events share one completion id.
        assert resolution["id"] == content["id"] == terminal["id"]

    @pytest.mark.asyncio
    async def test_include_usage_appends_one_usage_event_before_done(self):
        """include_usage -> resolution, content, terminal, usage, [DONE]. Exactly five."""
        events = _parse(
            [
                e
                async for e in _make_service().stream_response(
                    self._req(stream_options={"include_usage": True})
                )
            ]
        )

        assert len(events) == 5

        resolution, content, terminal, usage, done = events
        assert resolution["choices"] == []
        assert content["choices"][0]["delta"] == {"content": "ok"}
        assert terminal["choices"][0]["finish_reason"] == "stop"

        # The usage event carries an empty choices array, per OpenAI.
        assert usage["choices"] == []
        assert usage["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
        assert usage["id"] == content["id"]

        assert done == "[DONE]"

    @pytest.mark.asyncio
    async def test_nothing_is_emitted_after_the_usage_event(self):
        """DEC-053: usage is the last event before [DONE], on every path.

        Clients use the usage chunk as an end sentinel — OpenAI documents it as
        the chunk streamed before [DONE]. An event slipped in between would be
        dropped, or would terminate parsing early.
        """
        events = _parse(
            [
                e
                async for e in _make_service().stream_response(
                    self._req(stream_options={"include_usage": True})
                )
            ]
        )

        usage_index = next(
            i for i, e in enumerate(events) if isinstance(e, dict) and "usage" in e
        )
        assert events[usage_index + 1] == "[DONE]"
        assert usage_index == len(events) - 2

    @pytest.mark.asyncio
    async def test_resolution_precedes_every_content_event(self):
        """DEC-053: the pre-generation phase ends before the first token."""
        events = _parse(
            [e async for e in _make_service().stream_response(self._req())]
        )

        def _has_content(e) -> bool:
            return (
                isinstance(e, dict)
                and bool(e.get("choices"))
                and bool(e["choices"][0].get("delta", {}).get("content"))
            )

        resolution_index = next(
            i for i, e in enumerate(events) if isinstance(e, dict) and "resolved" in e
        )
        first_content = next(i for i, e in enumerate(events) if _has_content(e))
        assert resolution_index == 0
        assert resolution_index < first_content

        # Exactly one pre-generation event — cardinality is fixed (DEC-053).
        assert sum(1 for e in events if isinstance(e, dict) and "resolved" in e) == 1

    @pytest.mark.asyncio
    async def test_resolution_reports_the_clamped_value_not_the_requested_one(self):
        """`resolved` is built from the effective request, never the original."""
        events = _parse(
            [
                e
                async for e in _make_service().stream_response(
                    self._req(max_tokens=4000, max_context_tokens=200)
                )
            ]
        )

        resolution = events[0]
        assert resolution["resolved"]["max_tokens"] < 4000
        codes = {w["code"] for w in resolution["warnings"]}
        assert "max_tokens_clamped_to_context" in codes

    @pytest.mark.asyncio
    async def test_timeout_emits_error_then_done_and_nothing_else(self, monkeypatch):
        """Timeout -> resolution, error, [DONE]. No terminal event, no usage event.

        The pre-generation event has already gone out by the time the engine
        stalls — that is the point of putting it at the head rather than before
        [DONE] (DEC-053). It is the path where knowing what the server resolved
        matters most, and a trailer would have lost it.
        """

        class _HangingEngine(BaseEngine):
            async def generate(self, request):  # pragma: no cover - unused
                raise AssertionError("not used")

            async def generate_stream(self, request):
                import asyncio

                await asyncio.sleep(10)
                yield ChatStreamChunk(content="never")

            def is_healthy(self) -> bool:
                return True

        svc = _make_service(engine=_HangingEngine())

        class _Settings:
            stream_timeout_s = 0.01

        monkeypatch.setattr(
            "inference_x.services.chat_service.get_settings", lambda: _Settings()
        )

        events = _parse(
            [e async for e in svc.stream_response(self._req(stream_options={"include_usage": True}))]
        )

        assert len(events) == 3
        resolution, error, done = events
        assert "resolved" in resolution
        assert "error" in error
        assert done == "[DONE]"
        # include_usage was requested, but a timed-out stream accounted nothing.
        assert not any(isinstance(e, dict) and "usage" in e for e in events)


class TestReservationLifecycle:
    """fix-admission-reservation-leak: the reservation lifetime invariant.

    From the instant admit() successfully returns until the stream generator
    terminates for any reason, exactly one matching release() MUST occur, and
    a reservation MUST NEVER be released more than once. Each test below is
    one row of the termination matrix in
    openspec/changes/fix-admission-reservation-leak/design.md.
    """

    def _req(self, **kw) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
            **kw,
        )

    def _service(self, engine: BaseEngine) -> tuple[ChatService, _CountingAdmission]:
        registry = _make_registry("test")
        router = TaskRouter(registry, "test")
        admission = _CountingAdmission(AdmissionController(registry))
        pool = EnginePool({"test": engine})
        svc = ChatService(
            engine_pool=pool, registry=registry, router=router, admission=admission
        )
        return svc, admission

    def _assert_released_exactly_once_and_clear(self, admission: _CountingAdmission) -> None:
        assert len(admission.release_calls) == 1
        assert admission.inner._tracker.current("test") == 0
        assert admission.inner._seq_tracker.current("test") == 0

    @pytest.mark.asyncio
    async def test_normal_completion_releases_exactly_once(self):
        svc, admission = self._service(_StubEngine())
        events = [e async for e in svc.stream_response(self._req())]
        assert events[-1] == "data: [DONE]\n\n"
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_timeout_releases_exactly_once(self, monkeypatch):
        class _HangingEngine(BaseEngine):
            async def generate(self, request):  # pragma: no cover - unused
                raise AssertionError("not used")

            async def generate_stream(self, request):
                await asyncio.sleep(10)
                yield ChatStreamChunk(content="never")

            def is_healthy(self) -> bool:
                return True

        svc, admission = self._service(_HangingEngine())

        class _Settings:
            stream_timeout_s = 0.01

        monkeypatch.setattr(
            "inference_x.services.chat_service.get_settings", lambda: _Settings()
        )
        events = [e async for e in svc.stream_response(self._req())]
        assert events[-1] == "data: [DONE]\n\n"
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_engine_exception_releases_exactly_once(self):
        svc, admission = self._service(_StubEngine(raise_on_generate=True))
        with pytest.raises(RuntimeError, match="stub streaming error"):
            async for _ in svc.stream_response(self._req()):
                pass
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_cancelled_error_releases_exactly_once(self):
        class _HangingEngine(BaseEngine):
            async def generate(self, request):  # pragma: no cover - unused
                raise AssertionError("not used")

            async def generate_stream(self, request):
                yield ChatStreamChunk(content="ok")
                await asyncio.sleep(10)
                yield ChatStreamChunk(content="never")

            def is_healthy(self) -> bool:
                return True

        svc, admission = self._service(_HangingEngine())
        agen = svc.stream_response(self._req())
        await agen.__anext__()  # prologue
        await agen.__anext__()  # first content event
        task = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_generator_exit_before_first_token_releases_exactly_once(self):
        """The SEV-A1 regression: disconnect right after the DEC-053 prologue.

        Before the fix, this leaked the KV reservation and the sequence slot
        forever (reproduced during the Phase A audit as 522 KV tokens / 1
        sequence slot held after aclose()).
        """
        svc, admission = self._service(_StubEngine())
        agen = svc.stream_response(self._req())
        await agen.__anext__()  # prologue only — no content event yet
        await agen.aclose()
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_generator_exit_after_first_token_releases_exactly_once(self):
        svc, admission = self._service(_StubEngine())
        agen = svc.stream_response(self._req())
        await agen.__anext__()  # prologue
        await agen.__anext__()  # first content event
        await agen.aclose()
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_client_disconnect_after_terminal_event_releases_exactly_once(self):
        """Disconnect at the last suspension point before normal completion.

        Distinct from the "after first token" row: here the consumer has
        already received the terminal event and disconnects without ever
        requesting the event that would have triggered [DONE] — the point at
        which a real client's connection drop and this generator's cleanup are
        closest together.
        """
        svc, admission = self._service(_StubEngine())
        agen = svc.stream_response(self._req())
        await agen.__anext__()  # prologue
        await agen.__anext__()  # content
        await agen.__anext__()  # terminal
        await agen.aclose()
        self._assert_released_exactly_once_and_clear(admission)

    @pytest.mark.asyncio
    async def test_release_is_never_called_twice(self):
        """Closing an already-closed generator must not re-trigger release()."""
        svc, admission = self._service(_StubEngine())
        agen = svc.stream_response(self._req())
        await agen.__anext__()
        await agen.aclose()
        assert len(admission.release_calls) == 1
        await agen.aclose()  # no-op on an already-closed async generator
        assert len(admission.release_calls) == 1

    @pytest.mark.asyncio
    async def test_release_receives_the_reservation_admit_produced(self):
        """Identity, not just count: release() must receive the specific
        reservation admit() produced for THIS stream — not a coincidentally
        matching value, and not another concurrent stream's reservation.

        Exactly-once-release plus zero-tracker assertions (the other tests in
        this class) only prove value-correctness for a single reservation in
        isolation — under one outstanding reservation, the tracker returning
        to zero is only possible if release() used the same amount admit()
        added, so it can't be substituted for identity in general. It cannot
        catch a swap between two reservations that are simultaneously
        outstanding. This test makes that case impossible to pass by
        coincidence: two streams to the same model are admitted with
        deliberately different `max_tokens` (so their `reserved_tokens`
        differ), left concurrently outstanding, then closed independently —
        each release() call must carry its own stream's reservation.
        """
        svc, admission = self._service(_StubEngine())

        agen_small = svc.stream_response(self._req(max_tokens=16))
        await agen_small.__anext__()  # prologue — admit() has run for this stream
        agen_large = svc.stream_response(self._req(max_tokens=400))
        await agen_large.__anext__()  # prologue — admit() has run for this stream

        assert len(admission.admit_results) == 2
        reserved_small = admission.admit_results[0].reserved_tokens
        reserved_large = admission.admit_results[1].reserved_tokens
        # Not otherwise meaningful: if these coincided, a swap would be
        # unobservable and this test would pass regardless of correctness.
        assert reserved_small != reserved_large

        await agen_small.aclose()
        assert admission.release_calls == [("test", reserved_small)]

        await agen_large.aclose()
        assert admission.release_calls == [
            ("test", reserved_small),
            ("test", reserved_large),
        ]
