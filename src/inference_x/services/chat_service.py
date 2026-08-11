from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

from inference_x.benchmarks.hardware import extended_hardware_fields, profile_hardware
from inference_x.core.settings import get_settings
from inference_x.engines.base import BaseEngine
from inference_x.engines.pool import EnginePool
from inference_x.routing.admission import AdmissionController
from inference_x.routing.task_router import TaskRouter
from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamChunk,
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
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.ids import (
    compute_prompt_sha256,
    compute_run_id,
    is_hf_hub_repo_id,
    package_version,
    resolve_git_sha,
)

_RESOLVED_FIELDS = tuple(ResolvedRequest.model_fields)

_UNSET = object()
_manifest_hardware_cache: object = _UNSET


def _manifest_hardware_snapshot() -> ManifestHardware:
    """Cached for the process lifetime — hardware identity doesn't change
    between requests, and re-probing it per request (NVML init/shutdown,
    a /proc/cpuinfo scan) would add avoidable overhead to the very request
    path this platform exists to measure. Mirrors the `resolve_git_sha`
    caching pattern in `utils/ids.py`."""
    global _manifest_hardware_cache
    if _manifest_hardware_cache is not _UNSET:
        assert isinstance(_manifest_hardware_cache, ManifestHardware)
        return _manifest_hardware_cache
    hw_profile = profile_hardware()
    hw_extra = extended_hardware_fields()
    snapshot = ManifestHardware(
        gpu=hw_profile.gpu_name,
        vram_total_gib=hw_profile.vram_total_gb if hw_profile.has_gpu else None,
        driver=hw_extra.get("driver"),
        cuda=hw_extra.get("cuda"),
        cpu=hw_extra.get("cpu"),
        ram_gib=hw_profile.ram_total_gb,
        wsl2=hw_extra.get("wsl2"),
    )
    _manifest_hardware_cache = snapshot
    return snapshot


def _resolved(effective_request: ChatCompletionRequest) -> ResolvedRequest:
    """Serialize the Effective Request — what the server actually ran (OS-4).

    Built from *effective_request*, never from the client's original: building it
    from the original would report what was asked for rather than what ran, which
    inverts the point of the block.

    Field membership is derived from ``ResolvedRequest`` rather than listed here,
    so the derivability rule has exactly one home and this function cannot drift
    from it (see ResolvedRequest, and the schema test that enforces the rule).
    """
    return ResolvedRequest(
        **{name: getattr(effective_request, name) for name in _RESOLVED_FIELDS}
    )


def _batch_invariant_enabled() -> bool:
    """Whether VLLM_BATCH_INVARIANT is currently set — reported honestly."""
    from inference_x.utils.determinism import batch_invariant_env_enabled

    return batch_invariant_env_enabled()


def _build_manifest(
    *,
    routed_model: str,
    registry: ModelRegistry,
    engine: BaseEngine,
    original_request: ChatCompletionRequest,
    resolved: ResolvedRequest,
    warnings: list[ResponseWarning],
    response: ChatCompletionResponse,
) -> RunManifest:
    """Assemble the run manifest from signals the platform already produces.

    Phase C, C1 (add-run-manifest). Derives from *resolved*, *warnings*, and
    *response.timing/usage* — never recomputes token counts, timings, or
    warnings (design.md §"Goals"). Every provenance field the platform
    cannot determine is omitted/None, never fabricated (design.md D4).

    ``run_id``'s preimage is ``{engine, model, runtime, sampling, request,
    warnings}`` (design.md D2), where ``warnings`` is projected onto each
    warning's ``type``/``code``/``field`` only — never ``message``, which
    ``ResponseWarning``'s own contract (OS-4/DEC-053) permits to change
    without a spec change. ``timing``, ``batch``, and ``hardware`` are
    excluded from the hash entirely: ``timing``/``batch`` as
    speed/observational, ``hardware`` because ``run_id`` denotes
    configuration identity, not execution-environment identity (design.md
    D2) — ``hardware`` still rides the manifest as provenance.
    """
    model_entry = registry.get(routed_model)
    model_path = getattr(engine, "model_path", model_entry.model_path)

    manifest_engine = ManifestEngine(
        version=package_version("inferencex"),
        git_sha=resolve_git_sha(),
        backend_version=package_version("vllm"),
    )

    runtime_raw = getattr(engine, "runtime_snapshot", None)
    runtime_raw = runtime_raw if isinstance(runtime_raw, dict) else {}

    manifest_model = ManifestModel(
        registry_name=routed_model,
        hf_repo=model_path if is_hf_hub_repo_id(model_path) else None,
        quantization=model_entry.quantization,
        dtype=runtime_raw.get("dtype"),
    )

    manifest_runtime = ManifestRuntime(
        attention_backend=runtime_raw.get("attention_backend"),
        cuda_graphs=runtime_raw.get("cuda_graphs"),
        enforce_eager=runtime_raw.get("enforce_eager"),
        kv_cache_dtype=runtime_raw.get("kv_cache_dtype"),
        block_size=runtime_raw.get("block_size"),
        max_model_len=runtime_raw.get("max_model_len"),
        kv_capacity_tokens=getattr(engine, "kv_capacity_tokens", None),
        prefix_caching=runtime_raw.get("prefix_caching"),
        prefix_cache_hash_algo=runtime_raw.get("prefix_cache_hash_algo"),
        batch_invariant=_batch_invariant_enabled(),
    )

    manifest_sampling = ManifestSampling(
        temperature=resolved.temperature,
        top_p=resolved.top_p,
        seed=resolved.seed,
        max_tokens=original_request.max_tokens,
        resolved_max_tokens=resolved.max_tokens,
    )

    manifest_request = ManifestRequestInfo(
        prompt_sha256=compute_prompt_sha256(original_request.messages),
        prompt_tokens=response.usage.prompt_tokens,
        chat_template_sha256=getattr(engine, "chat_template_sha256", None),
    )

    manifest_hardware = _manifest_hardware_snapshot()

    preimage = {
        "engine": manifest_engine.model_dump(),
        "model": manifest_model.model_dump(),
        "runtime": manifest_runtime.model_dump(),
        "sampling": manifest_sampling.model_dump(),
        "request": manifest_request.model_dump(),
        # Stable identity fields only (design.md D2) — `message` is free text
        # that ResponseWarning's own contract (OS-4/DEC-053) already permits
        # to change without a spec change, so it must not affect run_id.
        "warnings": [
            {"type": w.type, "code": w.code, "field": w.field} for w in warnings
        ],
    }
    run_id = compute_run_id(preimage)

    return RunManifest(
        run_id=run_id,
        engine=manifest_engine,
        model=manifest_model,
        runtime=manifest_runtime,
        sampling=manifest_sampling,
        request=manifest_request,
        timing=response.timing,
        batch=ManifestBatch(),
        hardware=manifest_hardware,
        warnings=list(warnings),
    )


class ChatService:
    """Orchestrates chat completion requests through a router and engine pool.

    Phase 4 multi-model: the service dispatches to whichever engine in the pool
    matches the routed model name.  A pool with a single entry behaves identically
    to the previous single-engine setup.

    Phase 7/2 (VRAM-aware admission, see DEC-037): every request passes through
    an AdmissionController after routing and before dispatch, which enforces
    context-length limits and clamps/rejects under KV-pool pressure. Callers
    that construct ChatService directly (tests, stub engines) get a default
    controller with no VRAM tier — behaves like the pre-admission 4096-token cap.
    """

    def __init__(
        self,
        engine_pool: EnginePool,
        registry: ModelRegistry,
        router: TaskRouter,
        admission: AdmissionController | None = None,
    ) -> None:
        self._pool = engine_pool
        self._registry = registry
        self._router = router
        self._admission = admission or AdmissionController(registry)

    @staticmethod
    def _enforce_deterministic(request: ChatCompletionRequest) -> None:
        if request.deterministic:
            from inference_x.utils.determinism import ensure_deterministic_mode

            ensure_deterministic_mode(require=True)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route and execute a chat completion request.

        Raises:
            ValueError: routed model not in engine pool, or request rejected
                by admission control (context too long).
            EngineSaturatedError: KV pool saturated for a batch-tier request.
            RuntimeError: engine inference failure.
        """
        routed_model, engine = self._resolve_engine(request)
        self._enforce_deterministic(request)
        admitted = await self._admission.admit(routed_model, request, engine)
        effective_request = request.model_copy(
            update={"max_tokens": admitted.effective_max_tokens}
        )
        try:
            response = await engine.generate(effective_request)
        finally:
            self._admission.release(routed_model, admitted.reserved_tokens)
        # Attached here, never by the engine: the Engine Boundary does not learn
        # about admission (DEC-047).
        resolved = _resolved(effective_request)
        warnings = list(admitted.warnings)
        response = response.model_copy(
            update={"resolved": resolved, "warnings": warnings}
        )
        # Phase C, C1 (add-run-manifest): manifest assembly needs the fully
        # resolved response (usage.prompt_tokens, timing), so it runs after
        # the update above rather than before it.
        manifest = _build_manifest(
            routed_model=routed_model,
            registry=self._registry,
            engine=engine,
            original_request=request,
            resolved=resolved,
            warnings=warnings,
            response=response,
        )
        return response.model_copy(update={"run_id": manifest.run_id})

    async def stream_response(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Route a chat request and yield OpenAI-compatible SSE events.

        Event order is fixed (DEC-049 and OS-2 R3, extended at the head by
        DEC-053) and is the protocol:

        0. exactly one pre-generation event with ``choices: []`` carrying
           ``resolved`` and ``warnings``, always, before any content;
        1. zero or more content events, each with ``finish_reason: null``;
        2. exactly one terminal event with an empty delta and a real
           ``finish_reason``;
        3. one usage event with ``choices: []`` — only when the client asked via
           ``stream_options.include_usage`` and the engine accounted usage.
           Carries ``timing`` alongside ``usage`` (Phase B3) when the engine
           supplied it — one opt-in event for all terminal metadata, not a
           second flag;
        4. ``data: [DONE]``, always last.

        Event 0 sits at the head rather than before ``[DONE]`` because everything
        it carries is fixed the moment ``admit()`` returns, and a trailer would
        be lost on the timeout path — the path where knowing what the server
        resolved matters most (DEC-053). Nothing is ever emitted after the usage
        event: OpenAI documents that chunk as the one streamed before ``[DONE]``
        and clients use it as an end sentinel.

        The terminal event is separate rather than folded into the last content
        event, because the service cannot know a content event is the last one
        until the engine says so.

        Applies a per-token timeout (INFERENCE_X_STREAM_TIMEOUT_S) so that a
        stalled engine does not hold the connection open indefinitely. On
        timeout the error event is emitted and neither a terminal nor a usage
        event follows — only ``[DONE]``.

        Reservation lifetime: from the instant ``admit()`` returns until this
        generator terminates for any reason (normal completion, timeout, engine
        exception, cancellation, or the caller closing the generator — which is
        how a client disconnect surfaces here, including during the prologue
        event above), exactly one matching ``release()`` occurs. The ``try``
        below starts immediately after ``admit()`` and has a single ``finally``,
        so every suspension point in between — the prologue ``yield`` included —
        is covered by the same, single release call site.
        """
        routed_model, engine = self._resolve_engine(request)
        self._enforce_deterministic(request)
        admitted = await self._admission.admit(routed_model, request, engine)
        gen: AsyncGenerator[ChatStreamChunk, None] | None = None
        try:
            effective_request = request.model_copy(
                update={"max_tokens": admitted.effective_max_tokens}
            )
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            timeout_s = get_settings().stream_timeout_s
            include_usage = bool(
                request.stream_options and request.stream_options.include_usage
            )

            def _event(payload: dict) -> str:
                return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

            yield _event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "resolved": _resolved(effective_request).model_dump(),
                    "warnings": [w.model_dump() for w in admitted.warnings],
                }
            )

            gen = engine.generate_stream(effective_request)
            while True:
                try:
                    if timeout_s > 0:
                        chunk = await asyncio.wait_for(
                            gen.__anext__(), timeout=timeout_s
                        )
                    else:
                        chunk = await gen.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield (
                        f"data: {{\"error\":\"stream timed out after {timeout_s}s\"}}\n\n"
                    )
                    break

                if chunk.content:
                    yield _event(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "delta": {"content": chunk.content},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

                if chunk.finish_reason is not None:
                    yield _event(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": chunk.finish_reason,
                                }
                            ],
                        }
                    )
                    if include_usage and chunk.usage is not None:
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "usage": chunk.usage.model_dump(),
                        }
                        if chunk.timing is not None:
                            payload["timing"] = chunk.timing.model_dump()
                        yield _event(payload)
        finally:
            if gen is not None:
                await gen.aclose()
            self._admission.release(routed_model, admitted.reserved_tokens)

        yield "data: [DONE]\n\n"

    def _resolve_engine(self, request: ChatCompletionRequest) -> tuple[str, BaseEngine]:
        routed_model = self._router.select(request)

        loaded = self._pool.loaded_models()
        if routed_model not in loaded:
            loaded_str = loaded[0] if len(loaded) == 1 else str(loaded)
            raise ValueError(
                f"Routed to model '{routed_model}' but loaded model is "
                f"'{loaded_str}'. "
                f"Restart this process with INFERENCE_X_DEFAULT_MODEL={routed_model}."
            )

        return routed_model, self._pool.get(routed_model)

    def engine_healthy(self) -> bool:
        return self._pool.all_healthy()

    def loaded_models(self) -> list[str]:
        return self._pool.loaded_models()

    def registry(self) -> ModelRegistry:
        return self._registry
