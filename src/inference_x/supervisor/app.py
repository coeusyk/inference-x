"""The supervisor: one OpenAI endpoint in front of per-model workers (V1-2).

Start with `./scripts/dev.sh supervise` (or `uvicorn inference_x.supervisor.app:app`).
It never imports the worker app (`inference_x.api.main`), vLLM or torch, so it
holds no GPU memory itself. See add-model-lifecycle-supervisor design.md.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inference_x.api.routes.doctor import router as doctor_router
from inference_x.api.routes.models import router as models_router
from inference_x.api.routes.plan import router as plan_router
from inference_x.core.settings import get_settings
from inference_x.schemas.common import ErrorDetail, ErrorResponse
from inference_x.services.model_service import ModelRegistry
from inference_x.supervisor.workers import ModelBusy, ModelLoadFailed, Supervisor, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Headers a worker sets that clients rely on; everything else is hop-specific.
_PASSED_HEADERS = ("content-type", "x-run-id", "retry-after")


def _error(status: int, message: str, *, type_: str, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers=headers,
        content=ErrorResponse(error=ErrorDetail(message=message, type=type_, code=code)).model_dump(),
    )


def _build_supervisor() -> Supervisor:
    return Supervisor(
        max_loaded=int(os.environ.get("INFERENCE_X_MAX_LOADED_MODELS", "1")),
        switch_wait_s=float(os.environ.get("INFERENCE_X_SWITCH_WAIT_S", "60")),
        startup_timeout_s=float(os.environ.get("INFERENCE_X_WORKER_STARTUP_TIMEOUT_S", "600")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.supervisor = _build_supervisor()
    app.state.registry = ModelRegistry.from_config(get_settings().config_dir)
    logger.info("supervisor ready: models=%s", app.state.registry.names())
    yield
    await app.state.supervisor.shutdown()


app = FastAPI(title="InferenceX supervisor", lifespan=lifespan)
app.include_router(models_router, prefix="/v1")
app.include_router(plan_router, prefix="/v1")
app.include_router(doctor_router, prefix="/v1")


def _supervisor(request: Request) -> Supervisor:
    return request.app.state.supervisor  # type: ignore[no-any-return]


def _canonical(request: Request, model: Any) -> str | None:
    registry: ModelRegistry = request.app.state.registry
    return registry.canonical_name(model) if isinstance(model, str) else None


def _unknown_model(model: Any) -> JSONResponse:
    return _error(404, f"The model '{model}' is not registered.", type_="invalid_request_error", code="model_not_found")


async def _acquire(supervisor: Supervisor, model: str) -> Worker | JSONResponse:
    try:
        return await supervisor.acquire(model)
    except ModelBusy as exc:
        return _error(503, str(exc), type_="server_error", code="model_busy", headers={"Retry-After": "5"})
    except ModelLoadFailed as exc:
        return _error(503, str(exc), type_="server_error", code="model_load_failed")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    raw = await request.body()
    try:
        model = json.loads(raw).get("model")
    except (ValueError, AttributeError):
        return _error(400, "Request body must be a JSON object.", type_="invalid_request_error", code="invalid_json")
    canonical = _canonical(request, model)
    if canonical is None:
        return _unknown_model(model)

    supervisor = _supervisor(request)
    worker = await _acquire(supervisor, canonical)
    if isinstance(worker, JSONResponse):
        return worker

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = await client.send(
            client.build_request(
                "POST", f"{worker.url}/v1/chat/completions", content=raw,
                headers={"content-type": "application/json"},
            ),
            stream=True,
        )
    except httpx.TransportError as exc:
        await client.aclose()
        supervisor.release(worker)
        supervisor.reap(canonical)
        if not worker.alive():
            return _error(
                503, f"The worker for '{canonical}' stopped while serving the request.",
                type_="server_error", code="engine_unavailable",
            )
        return _error(502, f"Worker for '{canonical}' did not answer: {exc}", type_="server_error", code="worker_unreachable")

    async def body() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.TransportError:
            # The worker died mid-response; the status is already sent, so the
            # stream just ends. The next request finds it reaped (D5).
            logger.error("worker for %s stopped mid-response", canonical)
        finally:
            await upstream.aclose()
            await client.aclose()
            supervisor.release(worker)
            supervisor.reap(canonical)

    headers = {k: v for k, v in upstream.headers.items() if k.lower() in _PASSED_HEADERS}
    return StreamingResponse(body(), status_code=upstream.status_code, headers=headers)


@app.get("/v1/lifecycle")
async def lifecycle(request: Request) -> dict[str, Any]:
    supervisor = _supervisor(request)
    registry: ModelRegistry = request.app.state.registry
    return {"models": [supervisor.state(name) for name in registry.names()]}


async def _model_from_body(request: Request) -> tuple[str | None, Any]:
    try:
        model = (await request.json()).get("model")
    except (ValueError, AttributeError):
        model = None
    return _canonical(request, model), model


@app.post("/v1/lifecycle/load")
async def load(request: Request) -> Any:
    canonical, model = await _model_from_body(request)
    if canonical is None:
        return _unknown_model(model)
    supervisor = _supervisor(request)
    worker = await _acquire(supervisor, canonical)
    if isinstance(worker, JSONResponse):
        return worker
    supervisor.release(worker)
    return supervisor.state(canonical)


@app.post("/v1/lifecycle/unload")
async def unload(request: Request) -> Any:
    canonical, model = await _model_from_body(request)
    if canonical is None:
        return _unknown_model(model)
    supervisor = _supervisor(request)
    try:
        await supervisor.unload(canonical)
    except ModelBusy as exc:
        return _error(409, str(exc), type_="invalid_request_error", code="model_busy")
    return supervisor.state(canonical)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    supervisor = _supervisor(request)
    return {
        "status": "healthy",
        "loaded_models": [w.model for w in supervisor.loaded()],
    }
