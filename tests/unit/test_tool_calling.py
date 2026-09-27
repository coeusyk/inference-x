"""Streaming tool calling (add-streaming-tool-calling, DEC-065).

Request shapes are the ones Continue CLI 1.5.47 was captured sending. The
engine tests run the real installed vLLM Hermes parser, so a change in that
(unstable) API fails here rather than in production.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import inference_x.engines.vllm_engine as vllm_engine_module
from inference_x.api.deps import get_chat_service, get_registry
from inference_x.api.main import app
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.engines.vllm_engine import VLLMEngine, _held_back_start
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionUsage,
    ChatMessage,
    ChatStreamChunk,
    FunctionCall,
    ToolCall,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import (
    ChatService,
    ToolCallingUnsupportedError,
    _apply_parallel_tool_calls,
)
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.ids import compute_prompt_sha256, compute_tools_sha256

_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read the contents of a file at the specified path",
        "parameters": {
            "type": "object",
            "required": ["filepath"],
            "properties": {"filepath": {"type": "string", "description": "The path"}},
        },
    },
}
_CALL = {
    "id": "call_abc",
    "type": "function",
    "function": {"name": "Read", "arguments": '{"filepath":"hello.py"}'},
}
# The second request Continue sends after executing a tool.
_CONTINUE_BODY = {
    "model": "m",
    "stream": True,
    "stream_options": {"include_usage": True},
    "tools": [_READ_TOOL],
    "messages": [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "What does hello.py print?"},
        {"role": "assistant", "content": "", "tool_calls": [_CALL]},
        {"role": "tool", "content": 'print("hi")', "tool_call_id": "call_abc"},
    ],
}


def _request(**overrides) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**_CONTINUE_BODY, **overrides})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_continue_request_validates(self):
        req = _request()
        assert req.tools is not None and req.tools[0].function.name == "Read"
        assert req.messages[2].tool_calls[0].id == "call_abc"  # type: ignore[index]
        assert req.uses_tools

    def test_plain_request_does_not_use_tools(self):
        req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="x")])
        assert not req.uses_tools

    def test_tools_require_streaming(self):
        with pytest.raises(ValidationError, match="streaming"):
            _request(stream=False)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tool_choice": "auto"},
            {"parallel_tool_calls": "maybe"},
            {"tools": [{**_READ_TOOL, "function": {**_READ_TOOL["function"], "strict": True}}]},
            {"tools": [{"type": "code_interpreter", "function": _READ_TOOL["function"]}]},
            {"tools": []},
        ],
    )
    def test_unsupported_tool_options_rejected(self, overrides):
        with pytest.raises(ValidationError):
            _request(**overrides)

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "tool", "content": "x"},  # no tool_call_id
            {"role": "user", "content": "x", "tool_call_id": "a"},
            {"role": "user", "content": "x", "tool_calls": [_CALL]},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "function", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{**_CALL, "function": {"name": "Read", "arguments": "not json"}}],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{**_CALL, "function": {"name": "Read", "arguments": "[1]"}}],
            },
        ],
    )
    def test_malformed_tool_messages_rejected(self, message):
        with pytest.raises(ValidationError):
            ChatMessage.model_validate(message)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_plain_messages_keep_their_pre_tool_hash(self):
        msgs = [ChatMessage(role="user", content="hello")]
        # The exact pre-change canonical form: [{"role", "content"}] only.
        import hashlib

        legacy = hashlib.sha256(b'[{"content":"hello","role":"user"}]').hexdigest()
        assert compute_prompt_sha256(msgs) == legacy

    def test_tool_call_fields_change_the_prompt_hash(self):
        a = _request().messages
        b = [*a[:3], a[3].model_copy(update={"tool_call_id": "call_other"})]
        assert compute_prompt_sha256(a) != compute_prompt_sha256(b)

    def test_tools_hash_is_none_without_tools_and_tracks_definitions(self):
        assert compute_tools_sha256(None) is None
        tools = _request().tools
        other = _request(
            tools=[{**_READ_TOOL, "function": {**_READ_TOOL["function"], "description": "x"}}]
        ).tools
        assert compute_tools_sha256(tools) == compute_tools_sha256(_request().tools)
        assert compute_tools_sha256(tools) != compute_tools_sha256(other)


# ---------------------------------------------------------------------------
# Service: capability gate and SSE contract
# ---------------------------------------------------------------------------


class _ToolEngine(BaseEngine):
    def __init__(self, supports: bool, chunks: list[ChatStreamChunk] | None = None) -> None:
        self._supports = supports
        self._chunks = chunks or []
        self.calls = 0

    @property
    def supports_tools(self) -> bool:
        return self._supports

    async def generate(self, request):  # pragma: no cover - not used
        raise AssertionError

    async def generate_stream(self, request):
        self.calls += 1
        for chunk in self._chunks:
            yield chunk

    def is_healthy(self) -> bool:
        return True


def _service(engine: BaseEngine, model: str = "m") -> ChatService:
    registry = ModelRegistry([ModelEntry(name=model, model_path="test/m")])
    return ChatService(
        engine_pool=EnginePool({model: engine}),
        registry=registry,
        router=TaskRouter(registry, model),
    )


def _events(svc: ChatService, req: ChatCompletionRequest) -> list[object]:
    async def run() -> list[str]:
        return [e async for e in svc.stream_response(req)]

    out: list[object] = []
    for event in asyncio.run(run()):
        body = event.removeprefix("data: ").strip()
        out.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return out


_USAGE = ChatCompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class TestService:
    def test_unsupported_engine_rejects_before_admission(self):
        engine = _ToolEngine(supports=False)
        svc = _service(engine)
        with pytest.raises(ToolCallingUnsupportedError):
            _events(svc, _request())
        assert engine.calls == 0

    def test_tool_history_without_tools_also_needs_the_capability(self):
        svc = _service(_ToolEngine(supports=False))
        with pytest.raises(ToolCallingUnsupportedError):
            _events(svc, _request(tools=None, stream=False))

    def test_raw_malformed_arguments_reach_the_wire_unchanged(self):
        raw = '{"filepath": \'a.py\'}'
        call = ToolCall(id="call_1", function=FunctionCall(name="Read", arguments=raw))
        engine = _ToolEngine(
            supports=True,
            chunks=[ChatStreamChunk(finish_reason="tool_calls", tool_calls=[call])],
        )
        events = _events(_service(engine), _request())
        args_event = events[2]
        assert args_event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == raw  # type: ignore[index]

    def test_tool_call_stream_event_order(self):
        call = ToolCall(id="call_1", function=FunctionCall(name="Read", arguments='{"filepath": "a.py"}'))
        engine = _ToolEngine(
            supports=True,
            chunks=[
                ChatStreamChunk(content="Let me read it."),
                ChatStreamChunk(),  # heartbeat: no SSE event
                ChatStreamChunk(finish_reason="tool_calls", tool_calls=[call], usage=_USAGE),
            ],
        )
        events = _events(_service(engine), _request())
        prologue, content, identity, args, terminal, usage, done = events
        assert prologue["choices"] == [] and "tools" not in prologue["resolved"]  # type: ignore[index]
        assert content["choices"][0]["delta"] == {"content": "Let me read it."}  # type: ignore[index]
        assert identity["choices"][0]["delta"] == {  # type: ignore[index]
            "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "Read", "arguments": ""}}
            ]
        }
        assert identity["choices"][0]["finish_reason"] is None  # type: ignore[index]
        assert args["choices"][0]["delta"] == {  # type: ignore[index]
            "tool_calls": [{"index": 0, "function": {"arguments": '{"filepath": "a.py"}'}}]
        }
        assert terminal["choices"][0] == {"delta": {}, "index": 0, "finish_reason": "tool_calls"}  # type: ignore[index]
        assert usage["choices"] == [] and usage["usage"]["total_tokens"] == 15  # type: ignore[index]
        assert done == "[DONE]"

    def test_route_returns_openai_error_envelope(self):
        registry = ModelRegistry([ModelEntry(name="m", model_path="test/m")])
        app.dependency_overrides[get_chat_service] = lambda: _service(_ToolEngine(supports=False))
        app.dependency_overrides[get_registry] = lambda: registry
        try:
            with TestClient(app) as client:
                resp = client.post("/v1/chat/completions", json=_CONTINUE_BODY)
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 400
        assert resp.json()["error"] == {
            "message": "Model 'm' does not support tool calling: no tool-call parser is "
            "configured for it.",
            "type": "invalid_request_error",
            "param": "tools",
            "code": "tool_calling_unsupported",
        }


# ---------------------------------------------------------------------------
# vLLM adapter with the real Hermes parser
# ---------------------------------------------------------------------------


class _Completion:
    def __init__(self, text: str, finish_reason: str | None) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = [1] * max(1, len(text) // 4)


class _Output:
    def __init__(self, text: str, finished: bool) -> None:
        self.outputs = [_Completion(text, "stop" if finished else None)]
        self.finished = finished
        self.prompt_token_ids = [1, 2, 3]
        self.metrics = None


class _Tokenizer:
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.rendered: dict | None = None

    def apply_chat_template(self, messages, tools=None, **kwargs):
        self.rendered = {"messages": messages, "tools": tools}
        return "PROMPT"

    def get_vocab(self):
        return {}


class _LLM:
    errored = False

    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self.tokenizer = _Tokenizer()
        self.sampling = None

    def get_tokenizer(self):
        return self.tokenizer

    async def generate(self, prompt, sampling, request_id):
        self.sampling = sampling
        text = ""
        for i, delta in enumerate(self._deltas):
            text += delta
            yield _Output(text, finished=i == len(self._deltas) - 1)


class _Sampling:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture()
def _vllm_stubs(monkeypatch):
    pytest.importorskip("vllm.tool_parsers")
    monkeypatch.setattr(vllm_engine_module, "_VLLM_AVAILABLE", True)
    monkeypatch.setitem(vllm_engine_module.__dict__, "SamplingParams", _Sampling)


def _engine(deltas: list[str], parser: str | None = "hermes") -> VLLMEngine:
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._model_name = "m"
    engine._instruction_tuned = True
    engine._repetition_penalty = None
    engine._supports_chat = True
    engine._healthy = True
    engine._llm = _LLM(deltas)  # type: ignore[assignment]
    engine._tool_parser_cls = (
        vllm_engine_module._resolve_tool_parser(parser) if parser else None
    )
    return engine


def _chunks(engine: VLLMEngine, req: ChatCompletionRequest) -> list[ChatStreamChunk]:
    async def run():
        return [c async for c in engine.generate_stream(req)]

    return asyncio.run(run())


@pytest.mark.usefixtures("_vllm_stubs")
class TestVLLMAdapter:
    def test_supports_tools_follows_declared_parser(self):
        assert _engine([], "hermes").supports_tools
        assert not _engine([], None).supports_tools

    def test_unknown_parser_fails_at_resolution(self):
        with pytest.raises(RuntimeError, match="not provided"):
            vllm_engine_module._resolve_tool_parser("no-such-parser")

    def test_valid_call_streams_prefix_then_terminal_tool_call(self):
        engine = _engine(
            ["Reading", " it.<tool", '_call>\n{"name": "Read", ', '"arguments": {"filepath": "a.py"}}\n</tool_call>']
        )
        chunks = _chunks(engine, _request())
        content = "".join(c.content for c in chunks)
        assert content == "Reading it."
        terminal = chunks[-1]
        assert terminal.finish_reason == "tool_calls"
        assert terminal.tool_calls is not None and len(terminal.tool_calls) == 1
        call = terminal.tool_calls[0]
        assert call.function.name == "Read"
        assert json.loads(call.function.arguments) == {"filepath": "a.py"}
        assert call.id
        # Held-back steps produced heartbeats rather than nothing.
        assert any(c == ChatStreamChunk() for c in chunks)
        # Hermes parity: special tokens are kept so tags are not stripped.
        assert engine._llm.sampling.kwargs["skip_special_tokens"] is False  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("block", "raw_arguments"),
        [
            # Python-style quotes (seen live from Qwen3-4B through Continue's Edit tool).
            (
                '{"name": "Edit", "arguments": {"file_path": "s.py", "new_string": \'x = 1\\n\'}}',
                '{"file_path": "s.py", "new_string": \'x = 1\\n\'}',
            ),
            # Unbalanced nested array (seen live through Continue's MultiEdit tool).
            (
                '{"name": "MultiEdit", "arguments": {"edits": [{"old_string": "a"}}}',
                '{"edits": [{"old_string": "a"}}',
            ),
        ],
    )
    def test_recognized_call_with_invalid_arguments_is_transported_raw(self, block, raw_arguments):
        output = f"<tool_call>\n{block}\n</tool_call>"
        chunks = _chunks(_engine([output[:7], output[7:]]), _request())
        assert "".join(c.content for c in chunks) == ""
        terminal = chunks[-1]
        assert terminal.finish_reason == "tool_calls"
        assert terminal.tool_calls is not None and len(terminal.tool_calls) == 1
        call = terminal.tool_calls[0]
        assert block.startswith(f'{{"name": "{call.function.name}", ')
        # Byte for byte what the model wrote: no repair, no normalization.
        assert call.function.arguments == raw_arguments
        with pytest.raises(ValueError):
            json.loads(call.function.arguments)

    def test_valid_json_non_object_arguments_are_transported(self):
        output = '<tool_call>{"name": "Read", "arguments": "a.py"}</tool_call>'
        terminal = _chunks(_engine([output]), _request())[-1]
        assert terminal.finish_reason == "tool_calls"
        assert terminal.tool_calls[0].function.arguments == '"a.py"'  # type: ignore[index]

    @pytest.mark.parametrize(
        "output",
        [
            # Envelope never closed: which brace belongs to what is unknowable.
            '<tool_call>\n{"name": "Read", "arguments": {"filepath": "a.py"</tool_call>',
            # No arguments key.
            '<tool_call>{"name": "Read"}</tool_call>',
            # Name missing from the name slot.
            '<tool_call>{"arguments": {"filepath": "a.py"}}</tool_call>',
            # Empty name: valid JSON, but no function is identified.
            '<tool_call>{"name": "", "arguments": {}}</tool_call>',
            # Unclosed block with invalid arguments (e.g. cut off by max_tokens). An
            # unclosed block whose JSON is complete and valid is still a call (Hermes).
            '<tool_call>\n{"name": "Read", "arguments": {\'filepath\': "a.py"}}',
            # A second block that is not recognizable spoils the whole output.
            '<tool_call>{"name": "Read", "arguments": {}}</tool_call><tool_call>oops</tool_call>',
            # No tool-call framing at all (the Qwen2.5-Coder-1.5B failure).
            '```json\n{"name": "Read", "arguments": {"filepath": "a.py"}}\n```',
        ],
    )
    def test_unrecognizable_framing_is_returned_verbatim_never_as_a_call(self, output):
        chunks = _chunks(_engine([output[:5], output[5:]]), _request())
        assert all(c.tool_calls is None for c in chunks)
        assert "".join(c.content for c in chunks) == output
        assert chunks[-1].finish_reason == "stop"

    def test_tools_and_history_reach_the_chat_template(self):
        engine = _engine(["done"])
        _chunks(engine, _request())
        rendered = engine._llm.tokenizer.rendered  # type: ignore[attr-defined]
        assert rendered["tools"] == [_READ_TOOL]
        assistant, tool = rendered["messages"][2], rendered["messages"][3]
        assert assistant["tool_calls"][0]["function"]["arguments"] == {"filepath": "hello.py"}
        assert tool == {"role": "tool", "content": 'print("hi")', "tool_call_id": "call_abc"}

    def test_request_without_tools_is_unchanged(self):
        engine = _engine(["Hel", "lo <tool_call> literal"])
        req = ChatCompletionRequest(
            model="m", stream=True, messages=[ChatMessage(role="user", content="hi")]
        )
        chunks = _chunks(engine, req)
        assert [c.content for c in chunks[:-1]] == ["Hel", "lo <tool_call> literal"]
        assert chunks[-1].finish_reason == "stop" and chunks[-1].tool_calls is None
        assert "skip_special_tokens" not in engine._llm.sampling.kwargs  # type: ignore[attr-defined]


def test_held_back_start():
    tag = "<tool_call>"
    assert _held_back_start("abc", tag) == 3
    assert _held_back_start("abc<tool", tag) == 3
    assert _held_back_start("abc<tool_call>{", tag) == 3
    assert _held_back_start("a < b", tag) == 5


# ---------------------------------------------------------------------------
# Client-facing aliases (DEC-065)
# ---------------------------------------------------------------------------


class TestAliases:
    def _registry(self) -> ModelRegistry:
        return ModelRegistry(
            [ModelEntry(name="qwen3-4b-fp8", model_path="Qwen/Qwen3-4B", aliases=["local-4b"])]
        )

    def test_alias_routes_to_the_canonical_entry(self):
        registry = self._registry()
        assert registry.canonical_name("local-4b") == "qwen3-4b-fp8"
        assert registry.canonical_name("qwen3-4b-fp8") == "qwen3-4b-fp8"
        assert registry.canonical_name("unknown") is None
        router = TaskRouter(registry, "qwen3-4b-fp8")
        req = ChatCompletionRequest(model="local-4b", messages=[ChatMessage(role="user", content="x")])
        assert router.select(req) == "qwen3-4b-fp8"

    @pytest.mark.parametrize(
        "entries",
        [
            # Alias equal to another model's name.
            [("a", []), ("b", ["a"])],
            # Same alias on two models.
            [("a", ["x"]), ("b", ["x"])],
        ],
    )
    def test_alias_collisions_are_rejected(self, entries):
        with pytest.raises(ValueError, match="collides"):
            ModelRegistry(
                [ModelEntry(name=n, model_path=f"org/{n}", aliases=a) for n, a in entries]
            )

    def test_resolved_and_manifest_report_the_canonical_model(self):
        registry = self._registry()
        engine = _ToolEngine(
            supports=True, chunks=[ChatStreamChunk(finish_reason="stop", usage=_USAGE)]
        )
        svc = ChatService(
            engine_pool=EnginePool({"qwen3-4b-fp8": engine}),
            registry=registry,
            router=TaskRouter(registry, "qwen3-4b-fp8"),
        )
        prologue = _events(svc, _request(model="local-4b"))[0]
        assert prologue["resolved"]["model"] == "qwen3-4b-fp8"  # type: ignore[index]


# ---------------------------------------------------------------------------
# run_id identity: tools extend it only when used (DEC-065, design D8)
# ---------------------------------------------------------------------------

# run_id of `_golden_request()` computed by the pre-DEC-065 code on develop
# (4b2f797) with the same pinned environment. A no-tools request must keep it.
_PRE_DEC065_RUN_ID = "sha256:ec6cba87880269912a3485c668b15da12877241fa2332ccfcd0563a15a5d02fb"


def _golden_request(**overrides) -> ChatCompletionRequest:
    body = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.0,
        "seed": 7,
        "stop": ["\n\n"],
        **overrides,
    }
    return ChatCompletionRequest.model_validate(body)


@pytest.fixture()
def _pinned_manifest_env(monkeypatch):
    import inference_x.services.chat_service as cs
    from inference_x.schemas.chat import ManifestHardware

    monkeypatch.setattr(cs, "package_version", lambda name: f"{name}-1.0")
    monkeypatch.setattr(cs, "resolve_git_sha", lambda: "0123abcd")
    monkeypatch.setattr(cs, "_batch_invariant_enabled", lambda: False)
    monkeypatch.setattr(cs, "_manifest_hardware_snapshot", lambda: ManifestHardware())


def _manifest(req: ChatCompletionRequest):
    from inference_x.schemas.chat import (
        ChatCompletionChoice,
        ChatCompletionMessage,
        ChatCompletionResponse,
    )
    from inference_x.services.chat_service import _build_manifest, _resolved

    class _Engine:
        model_path = "org/model"

    response = ChatCompletionResponse(
        model="test",
        choices=[
            ChatCompletionChoice(
                index=0, message=ChatCompletionMessage(content="ok"), finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=12, completion_tokens=1, total_tokens=13),
    )
    return _build_manifest(
        routed_model="test",
        registry=ModelRegistry([ModelEntry(name="test", model_path="org/model")]),
        engine=_Engine(),  # type: ignore[arg-type]
        original_request=req,
        resolved=_resolved(req.model_copy(update={"max_tokens": 100})),
        warnings=[],
        response=response,
    )


def _documented_preimage(manifest) -> dict:
    """The preimage rule as documented (add-run-manifest D2 + DEC-065)."""
    request = manifest.request.model_dump()
    if request["tools_sha256"] is None:
        del request["tools_sha256"]
    return {
        "engine": manifest.engine.model_dump(),
        "model": manifest.model.model_dump(),
        "runtime": manifest.runtime.model_dump(),
        "sampling": manifest.sampling.model_dump(),
        "request": request,
        "warnings": [{"type": w.type, "code": w.code, "field": w.field} for w in manifest.warnings],
    }


@pytest.mark.usefixtures("_pinned_manifest_env")
class TestRunIdentity:
    # The golden fixture is non-streaming, and `tools` requires streaming at the
    # schema level. model_construct sets `tools` on an otherwise validated request
    # so the manifest rule can be exercised for when that restriction lifts.
    def _with_tools(self, tools: list[dict]) -> ChatCompletionRequest:
        from inference_x.schemas.chat import ToolDefinition

        req = _golden_request()
        req.__dict__["tools"] = [ToolDefinition.model_validate(t) for t in tools]
        return req

    def test_no_tools_run_id_is_unchanged_from_before_dec_065(self):
        manifest = _manifest(_golden_request())
        assert manifest.run_id == _PRE_DEC065_RUN_ID
        # The manifest may still expose the key, as null.
        assert manifest.request.tools_sha256 is None

    def test_adding_tools_changes_the_run_id(self):
        with_tools = _manifest(self._with_tools([_READ_TOOL]))
        assert with_tools.request.tools_sha256 is not None
        assert with_tools.run_id != _PRE_DEC065_RUN_ID

    def test_changing_only_tool_definitions_changes_the_run_id(self):
        other = {**_READ_TOOL, "function": {**_READ_TOOL["function"], "description": "other"}}
        a = _manifest(self._with_tools([_READ_TOOL]))
        b = _manifest(self._with_tools([other]))
        assert a.run_id != b.run_id

    def test_key_order_inside_schemas_does_not_change_tools_sha256(self):
        reordered = {
            "function": {
                "parameters": {
                    "properties": {"filepath": {"description": "The path", "type": "string"}},
                    "required": ["filepath"],
                    "type": "object",
                },
                "description": _READ_TOOL["function"]["description"],
                "name": "Read",
            },
            "type": "function",
        }
        a = _manifest(self._with_tools([_READ_TOOL]))
        b = _manifest(self._with_tools([reordered]))
        assert a.request.tools_sha256 == b.request.tools_sha256
        assert a.run_id == b.run_id

    @pytest.mark.parametrize("tools", [None, [_READ_TOOL]])
    def test_run_id_is_recomputable_from_the_documented_preimage(self, tools):
        from inference_x.utils.ids import compute_run_id

        req = _golden_request() if tools is None else self._with_tools(tools)
        manifest = _manifest(req)
        assert compute_run_id(_documented_preimage(manifest)) == manifest.run_id


# ---------------------------------------------------------------------------
# parallel_tool_calls (DEC-065): Continue for VS Code always sends false
# ---------------------------------------------------------------------------


def _call(i: int, arguments: str = '{"filepath": "a.py"}') -> ToolCall:
    return ToolCall(id=f"call_{i}", function=FunctionCall(name="Read", arguments=arguments))


def _stream_calls(calls: list[ToolCall], **overrides) -> tuple[list, dict, dict]:
    engine = _ToolEngine(
        supports=True,
        chunks=[ChatStreamChunk(finish_reason="tool_calls", tool_calls=calls, usage=_USAGE)],
    )
    events = _events(_service(engine), _request(**overrides))
    prologue = events[0]
    terminal = next(
        e for e in events[1:]
        if isinstance(e, dict) and e["choices"] and e["choices"][0]["finish_reason"]
    )
    identities = [
        e["choices"][0]["delta"]["tool_calls"][0]
        for e in events[1:]
        if isinstance(e, dict) and e["choices"]
        and "id" in e["choices"][0]["delta"].get("tool_calls", [{}])[0]
    ]
    return identities, prologue, terminal  # type: ignore[return-value]


class TestParallelToolCalls:
    @pytest.mark.parametrize("overrides", [{}, {"parallel_tool_calls": True}])
    def test_absent_or_true_transports_every_call(self, overrides):
        identities, prologue, terminal = _stream_calls([_call(0), _call(1)], **overrides)
        assert [c["id"] for c in identities] == ["call_0", "call_1"]
        assert "warnings" not in terminal
        assert prologue["resolved"]["parallel_tool_calls"] == overrides.get("parallel_tool_calls")

    def test_false_with_one_call_is_normal(self):
        identities, prologue, terminal = _stream_calls([_call(0)], parallel_tool_calls=False)
        assert [c["id"] for c in identities] == ["call_0"]
        assert "warnings" not in terminal
        assert prologue["resolved"]["parallel_tool_calls"] is False

    def test_false_with_two_calls_returns_the_first_and_discloses_it(self):
        identities, _, terminal = _stream_calls([_call(0), _call(1)], parallel_tool_calls=False)
        assert [c["id"] for c in identities] == ["call_0"]
        assert terminal["choices"][0]["finish_reason"] == "tool_calls"
        assert terminal["warnings"] == [
            {
                "type": "substituted",
                "code": "parallel_tool_calls_truncated",
                "message": "The model produced 2 tool calls while parallel_tool_calls=false; "
                "only the first call was returned.",
                "field": "parallel_tool_calls",
            }
        ]


@pytest.mark.usefixtures("_vllm_stubs")
class TestFirstCallIsIndependentOfTheSecond:
    _FIRST = '<tool_call>\n{"name": "Read", "arguments": {"filepath":"a.py",  "n": 1.0}}\n</tool_call>'

    @pytest.mark.parametrize(
        "second",
        [
            '<tool_call>\n{"name": "Read", "arguments": {"filepath": "b.py"}}\n</tool_call>',
            "<tool_call>\n{\"name\": \"Read\", \"arguments\": {'filepath': 'b.py'}}\n</tool_call>",
        ],
    )
    def test_malformed_second_call_does_not_alter_the_first(self, second):
        terminal = _chunks(_engine([self._FIRST, second]), _request())[-1]
        assert terminal.tool_calls is not None and len(terminal.tool_calls) == 2
        # The model's own bytes, whether the neighbour parsed or not (no
        # json.dumps re-spacing, no 1.0 -> 1.0 rewriting, nothing).
        assert terminal.tool_calls[0].function.arguments == '{"filepath":"a.py",  "n": 1.0}'

    def test_truncation_keeps_the_first_call_byte_for_byte(self):
        second = "<tool_call>\n{\"name\": \"Read\", \"arguments\": {'x': 1}}\n</tool_call>"
        calls = _chunks(_engine([self._FIRST, second]), _request())[-1].tool_calls
        kept, warnings = _apply_parallel_tool_calls(_request(parallel_tool_calls=False), calls)
        assert [c.function.arguments for c in kept] == ['{"filepath":"a.py",  "n": 1.0}']
        assert [w.code for w in warnings] == ["parallel_tool_calls_truncated"]
