"""The llama.cpp backend's wiring outside the engine (add-llama-cpp-backend):
config validation, factory dispatch, provenance, plan/doctor, errors."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_x.api.deps import get_chat_service, get_registry, get_vram_tier
from inference_x.api.main import app
from inference_x.engines.base import BaseEngine, EngineUnavailableError
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from inference_x.schemas.model import ModelEntry
from inference_x.services.chat_service import _build_manifest, _resolved
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.ids import compute_run_id

_GGUF = {"name": "g", "engine": "llama_cpp", "model_path": "org/repo-GGUF",
         "gguf_file": "m-q4_k_m.gguf", "max_model_len": 8192}


# Config ----------------------------------------------------------------------


def test_llama_entry_defaults_to_one_sequence():
    assert ModelEntry(**_GGUF).max_num_seqs == 1


@pytest.mark.parametrize(
    "override, message",
    [
        ({"gpu_memory_utilization": 0.8}, "gpu_memory_utilization"),
        ({"quantization": "awq"}, "quantization"),
        ({"tool_call_parser": "hermes"}, "tool_call_parser"),
        ({"max_num_batched_tokens": 512}, "max_num_batched_tokens"),
        ({"max_num_seqs": 2}, "one sequence per process"),
        ({"max_model_len": None}, "requires max_model_len"),
        ({"gguf_file": None}, "local .gguf model_path"),
        ({"model_path": "local/m.gguf"}, "local .gguf model_path"),
    ],
)
def test_llama_entry_rejects_what_it_would_ignore(override, message):
    with pytest.raises(ValidationError, match=message):
        ModelEntry(**{**_GGUF, **override})


def test_local_gguf_path_needs_no_gguf_file():
    entry = ModelEntry(name="l", engine="llama_cpp", model_path="models/m.gguf", max_model_len=2048)
    assert entry.gguf_file is None


@pytest.mark.parametrize("field, value", [("gguf_file", "x.gguf"), ("n_gpu_layers", 10)])
def test_vllm_entry_rejects_llama_only_fields(field, value):
    with pytest.raises(ValidationError, match="apply only to engine: llama_cpp"):
        ModelEntry(name="v", model_path="org/m", **{field: value})


# Factory -----------------------------------------------------------------------


def test_factory_builds_llama_engine_without_vllm_sizing(monkeypatch):
    built: list[dict] = []

    class _Recording:
        def __init__(self, config: dict) -> None:
            built.append(config)

    monkeypatch.setattr("inference_x.engines.llama_cpp_engine.LlamaCppEngine", _Recording)

    def _no_vllm_sizing(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("vLLM sizing ran for a llama.cpp entry")

    monkeypatch.setattr("inference_x.utils.vllm_pool_config.validate_model_fits", _no_vllm_sizing)
    monkeypatch.setattr("inference_x.utils.vllm_pool_config.apply_tier_knobs", _no_vllm_sizing)

    from inference_x.engines.registry import create_engine

    create_engine(ModelEntry(**_GGUF), None, 4.0, 8.0)
    assert built and built[0]["gguf_file"] == "m-q4_k_m.gguf"


def test_deterministic_startup_refuses_llama_models(monkeypatch, tmp_path):
    import inference_x.api.deps as deps
    from inference_x.core.settings import get_settings

    (tmp_path / "models.yaml").write_text(
        "models:\n  - name: g\n    engine: llama_cpp\n    model_path: org/repo-GGUF\n"
        "    gguf_file: m.gguf\n    max_model_len: 8192\n"
    )
    monkeypatch.setenv("INFERENCE_X_DETERMINISTIC", "1")
    get_settings.cache_clear()
    deps._build_registry.cache_clear()
    deps._build_engine_pool.cache_clear()
    monkeypatch.setattr(deps, "_resolve_vram_tier_for_pool", lambda config_dir: None)
    monkeypatch.setattr(deps, "probe_gpu_memory_gib", lambda: (4.0, 8.0))

    def _must_not_build(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("engine built despite deterministic mode")

    monkeypatch.setattr(deps, "create_engine", _must_not_build)
    try:
        with pytest.raises(ValueError, match="only supported for vLLM models"):
            deps._build_engine_pool(str(tmp_path), ("g",))
    finally:
        get_settings.cache_clear()
        deps._build_registry.cache_clear()
        deps._build_engine_pool.cache_clear()


# Provenance --------------------------------------------------------------------


class _GgufEngine(BaseEngine):
    model_path = "org/repo-GGUF"
    chat_template_sha256 = "t" * 64
    kv_capacity_tokens = 8192
    manifest_identity = {
        "backend": "llama.cpp",
        "backend_version": "b11217-c9064dded",
        "hf_repo": "org/repo-GGUF",
        "hf_revision": "abc123",
        "gguf_file": "m-q4_k_m.gguf",
        "weights_sha256": "f" * 64,
        "quantization": "Q4_K - Medium",
        "hardware_cuda": None,
    }
    runtime_snapshot = {"max_model_len": 8192, "kv_cache_dtype": "f16",
                        "prefix_caching": False, "batch_invariant": False}

    async def generate(self, request):  # pragma: no cover - not called
        raise NotImplementedError

    def generate_stream(self, request):  # pragma: no cover - not called
        raise NotImplementedError

    def is_healthy(self) -> bool:
        return True


def _manifest(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")  # must not leak into a llama.cpp manifest
    registry = ModelRegistry([ModelEntry(**_GGUF)])
    request = ChatCompletionRequest(
        model="g", messages=[ChatMessage(role="user", content="hi")], max_tokens=4
    )
    resolved = _resolved(request)
    response = ChatCompletionResponse(
        model="g",
        choices=[ChatCompletionChoice(index=0, message=ChatCompletionMessage(content="x"), finish_reason="stop")],
        usage=ChatCompletionUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        resolved=resolved,
    )
    return _build_manifest(
        routed_model="g", registry=registry, engine=_GgufEngine(), original_request=request,
        resolved=resolved, warnings=[], response=response,
    )


def test_manifest_reports_backend_and_gguf_identity(monkeypatch):
    manifest = _manifest(monkeypatch)
    assert manifest.engine.backend == "llama.cpp"
    assert manifest.engine.backend_version == "b11217-c9064dded"
    assert manifest.model.model_dump() == {
        "registry_name": "g", "hf_repo": "org/repo-GGUF", "hf_revision": "abc123",
        "weights_sha256": "f" * 64, "quantization": "Q4_K - Medium", "dtype": None,
        "gguf_file": "m-q4_k_m.gguf",
    }
    runtime = manifest.runtime
    assert (runtime.max_model_len, runtime.kv_capacity_tokens) == (8192, 8192)
    assert runtime.kv_cache_dtype == "f16" and runtime.prefix_caching is False
    assert runtime.batch_invariant is False
    assert runtime.cuda_graphs is None and runtime.block_size is None
    assert manifest.hardware.cuda is None
    assert manifest.timing is None


def test_gguf_file_is_part_of_run_identity(monkeypatch):
    manifest = _manifest(monkeypatch)
    preimage = {
        "engine": manifest.engine.model_dump(),
        "model": manifest.model.model_dump(),
        "runtime": manifest.runtime.model_dump(),
        "sampling": manifest.sampling.model_dump(),
        "request": manifest.request.model_dump(exclude={"tools_sha256"}),
        "warnings": [],
    }
    assert compute_run_id(preimage) == manifest.run_id


# Plan, doctor, models, errors ----------------------------------------------------


@pytest.fixture
def gguf_client(monkeypatch):
    registry = ModelRegistry([ModelEntry(**_GGUF)])
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_vram_tier] = lambda: None
    yield monkeypatch
    app.dependency_overrides.clear()


def _cached(monkeypatch, present: bool, size: float = 4.36) -> None:
    from pathlib import Path

    for module in ("plan", "doctor"):
        monkeypatch.setattr(
            f"inference_x.api.routes.{module}.cached_gguf_path",
            lambda *a: Path("m.gguf") if present else None,
        )
    monkeypatch.setattr("inference_x.api.routes.plan.gguf_size_gib", lambda *a: size if present else None)
    monkeypatch.setattr("inference_x.api.routes.models.gguf_size_gib", lambda *a: size if present else None)


def _fit(monkeypatch, layers: int | None, error: str | None = None) -> None:
    for module in ("plan", "doctor"):
        monkeypatch.setattr(f"inference_x.api.routes.{module}.fit_gpu_layers", lambda *a: (layers, error))


def test_plan_uses_llama_fit_and_nulls_vllm_estimates(gguf_client):
    _cached(gguf_client, True)
    _fit(gguf_client, -1)
    with TestClient(app) as c:
        entry = c.get("/v1/plan").json()["models"][0]
    assert entry["backend"] == "llama_cpp"
    assert entry["estimated_weight_gib"] == 4.36
    assert entry["gpu_layers"] == -1
    for vllm_only in ("estimated_kv_cache_gib", "estimated_footprint_gib", "gpu_memory_utilization",
                      "block_size", "kv_cache_dtype", "max_num_batched_tokens"):
        assert entry[vllm_only] is None


@pytest.mark.parametrize(
    "present, layers, error, fits, reason",
    [
        (True, -1, None, True, None),
        (True, 14, None, False, "only 14 layers fit"),
        (True, None, "llama-server not found", False, "llama-server not found"),
        (False, None, None, False, "GGUF file not on disk"),
    ],
)
def test_doctor_reports_llama_fit(gguf_client, present, layers, error, fits, reason):
    _cached(gguf_client, present)
    _fit(gguf_client, layers, error)
    with TestClient(app) as c:
        entry = c.get("/v1/doctor").json()["models"][0]
    assert entry["backend"] == "llama_cpp"
    assert entry["fits"] is fits
    if reason is None:
        assert entry["reason"] is None
    else:
        assert reason in entry["reason"]


def test_models_reports_gguf_size_or_null(gguf_client):
    _cached(gguf_client, False)
    with TestClient(app) as c:
        assert c.get("/v1/models").json()["data"][0]["estimated_weights_gib"] is None


def test_dead_backend_is_503_engine_unavailable():
    class _DeadService:
        async def complete(self, request):
            raise EngineUnavailableError("llama-server for g is not running (exit code -9)")

    app.dependency_overrides[get_chat_service] = lambda: _DeadService()
    try:
        with TestClient(app) as c:
            resp = c.post("/v1/chat/completions", json={"model": "g", "messages": [{"role": "user", "content": "hi"}]})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "engine_unavailable"
