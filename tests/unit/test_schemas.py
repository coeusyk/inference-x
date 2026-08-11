"""Unit tests for chat schemas."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_x.api.deps import get_chat_service, get_registry
from inference_x.api.main import app
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ManifestBatch,
    ManifestEngine,
    ManifestHardware,
    ManifestModel,
    ManifestRequestInfo,
    ManifestRuntime,
    ManifestSampling,
    ResolvedRequest,
    ResponseWarning,
    RunManifest,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import ChatService
from inference_x.services.model_service import ModelRegistry

_TEST_MODEL = "test-model"


class _StubEngine(BaseEngine):
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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
        yield "ok"

    def is_healthy(self) -> bool:
        return True


def _registry() -> ModelRegistry:
    return ModelRegistry([ModelEntry(name=_TEST_MODEL, model_path="test/stub")])


def _service() -> ChatService:
    registry = _registry()
    router = TaskRouter(registry, _TEST_MODEL)
    pool = EnginePool({_TEST_MODEL: _StubEngine()})
    return ChatService(engine_pool=pool, registry=registry, router=router)


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

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                temperature=-0.1,
            )

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=0,
            )

    def test_max_tokens_exceeds_4096_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=4097,
            )

    def test_content_exceeds_max_length_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="x" * 32001)

    def test_content_at_max_length_is_valid(self):
        msg = ChatMessage(role="user", content="x" * 32000)
        assert len(msg.content) == 32000

    def test_temperature_at_boundaries_is_valid(self):
        req_min = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=0.0,
        )
        req_max = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            temperature=2.0,
        )
        assert req_min.temperature == 0.0
        assert req_max.temperature == 2.0

    def test_max_tokens_at_boundary_is_valid(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            max_tokens=4096,
        )
        assert req.max_tokens == 4096

    def test_seed_integer_is_accepted(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            seed=42,
        )
        assert req.seed == 42

    def test_seed_minus_one_is_accepted(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
            seed=-1,
        )
        assert req.seed == -1

    def test_seed_defaults_to_none(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role="user", content="x")],
        )
        assert req.seed is None

    def test_seed_non_integer_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="x")],
                seed="not-an-int",  # type: ignore[arg-type]
            )


class TestSchemaConstraintsViaAPI:
    """Verify that out-of-range values produce HTTP 422 when sent through the API."""

    def _client(self) -> TestClient:
        app.dependency_overrides[get_chat_service] = _service
        app.dependency_overrides[get_registry] = _registry
        return TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.clear()

    def _base_payload(self) -> dict:
        return {
            "model": _TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
        }

    def test_temperature_above_2_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["temperature"] = 3.0
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_temperature_below_0_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["temperature"] = -0.5
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_max_tokens_above_4096_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["max_tokens"] = 5000
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_max_tokens_zero_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["max_tokens"] = 0
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_content_exceeds_32000_chars_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["messages"] = [{"role": "user", "content": "x" * 32001}]
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_valid_request_returns_200(self):
        client = self._client()
        resp = client.post("/v1/chat/completions", json=self._base_payload())
        assert resp.status_code == 200

    def test_seed_integer_returns_200(self):
        client = self._client()
        payload = self._base_payload()
        payload["seed"] = 42
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    def test_seed_non_integer_returns_422(self):
        client = self._client()
        payload = self._base_payload()
        payload["seed"] = "nope"
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422


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

    def test_run_id_defaults_to_none(self):
        assert self._make_response().run_id is None


class TestEffectiveRequestSurfaces:
    """OS-4: `resolved`, `warnings` and `strict` — the wire shape of substitution."""

    # The two exclusion classes from the derivability rule: message content, and
    # the transport/policy controls. Everything else must appear in `resolved`.
    _EXCLUDED = {"messages", "stream", "stream_options", "strict"}

    def test_resolved_field_set_is_derivable_from_the_request(self):
        """The rule is enforced here rather than in review.

        A field added to ChatCompletionRequest later fails this test, forcing an
        explicit decision about whether it belongs in `resolved` — which is the
        only thing stopping the block becoming a curated junk drawer.
        """
        assert set(ResolvedRequest.model_fields) == (
            set(ChatCompletionRequest.model_fields) - self._EXCLUDED
        )

    def test_strict_defaults_to_false_and_is_last(self):
        assert ChatCompletionRequest.model_fields["strict"].default is False
        assert list(ChatCompletionRequest.model_fields)[-1] == "strict"

    def test_non_boolean_strict_is_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="hi")],
                strict="yes-please",
            )

    def test_response_defaults_are_additive(self):
        """C1/C2: existing construction keeps working, and `warnings` is empty
        rather than absent — an absent collection reads as 'not implemented'."""
        resp = ChatCompletionResponse(
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        assert resp.resolved is None
        assert resp.warnings == []
        assert "warnings" in resp.model_dump()

    def test_warning_type_is_a_closed_two_member_set(self):
        for ok in ("substituted", "degraded"):
            assert ResponseWarning(type=ok, code="c", message="m").type == ok
        with pytest.raises(ValidationError):
            ResponseWarning(type="informational", code="c", message="m")


class TestRunManifest:
    """Phase C, C1 (add-run-manifest) manifest block shapes/defaults."""

    def _manifest(self, **overrides) -> RunManifest:
        payload = {
            "run_id": "sha256:" + "a" * 64,
            "engine": ManifestEngine(),
            "model": ManifestModel(registry_name="test"),
            "runtime": ManifestRuntime(),
            "sampling": ManifestSampling(),
            "request": ManifestRequestInfo(),
            "hardware": ManifestHardware(),
        }
        payload.update(overrides)
        return RunManifest(**payload)

    def test_manifest_version_defaults_to_1(self):
        assert self._manifest().manifest_version == 1

    def test_batch_defaults_to_empty_co_batched_request_ids(self):
        manifest = self._manifest()
        assert manifest.batch.co_batched_request_ids == []

    def test_timing_and_warnings_default_to_absent_or_empty(self):
        manifest = self._manifest()
        assert manifest.timing is None
        assert manifest.warnings == []

    def test_model_hf_revision_and_weights_sha256_default_to_none(self):
        model = ManifestModel(registry_name="test")
        assert model.hf_revision is None
        assert model.weights_sha256 is None

    def test_runtime_batch_invariant_defaults_to_false(self):
        assert ManifestRuntime().batch_invariant is False

    def test_hardware_fields_default_to_none(self):
        hw = ManifestHardware()
        assert hw.gpu is None
        assert hw.wsl2 is None

    def test_manifest_serializes_to_dict_with_all_blocks(self):
        data = self._manifest().model_dump()
        for block in ("engine", "model", "runtime", "sampling", "request", "batch", "hardware"):
            assert block in data
