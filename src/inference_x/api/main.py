import logging
import logging.config
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from starlette.routing import Mount

from inference_x.utils.vllm_platform_patch import apply as apply_vllm_platform_patch

# vLLM worker env and platform defaults before any worker/subprocess imports vLLM.
from inference_x.utils.cuda_env import ensure_vllm_process_env

ensure_vllm_process_env()
apply_vllm_platform_patch()

from inference_x.api import deps
from inference_x.api.errors import (
    engine_saturated_error_handler,
    runtime_error_handler,
    value_error_handler,
)
from inference_x.api.routes.benchmark import router as benchmark_router
from inference_x.api.routes.chat_completions import router as chat_router
from inference_x.api.routes.health import router as health_router
from inference_x.api.routes.metrics import router as metrics_router
from inference_x.api.routes.models import router as models_router
from inference_x.observability.middleware import ObservabilityMiddleware
from inference_x.routing.admission import EngineSaturatedError


def _configure_logging() -> None:
    config_dir = Path(os.environ.get("INFERENCE_X_CONFIG_DIR", "config"))
    logging_path = config_dir / "logging.yaml"
    if logging_path.is_file():
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        with logging_path.open(encoding="utf-8") as fh:
            logging.config.dictConfig(yaml.safe_load(fh))
    else:
        logging.basicConfig(level=logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize registry, router, and engine before serving requests."""
    try:
        deps.initialize_app()
    except Exception as exc:
        logger.critical("Startup initialization failed: %s", exc, exc_info=True)
        raise
    yield
    deps.shutdown_app()


app = FastAPI(
    title="InferenceX",
    description="Self-hosted vLLM-backed OpenAI-compatible inference API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(RuntimeError, runtime_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(EngineSaturatedError, engine_saturated_error_handler)  # type: ignore[arg-type]

# Observability middleware — must be added before routers so it wraps all paths.
app.add_middleware(ObservabilityMiddleware, recorder=deps.get_recorder())

app.include_router(health_router)
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(metrics_router, prefix="/v1")
app.include_router(benchmark_router, prefix="/v1/benchmark")

# vLLM's own native Prometheus stat logger, mounted as-is against the default
# prometheus_client.REGISTRY it already registers into (verified empirically —
# see openspec/changes/expose-native-engine-metrics/design.md Decision 1).
# Distinct from GET /v1/metrics (InferenceX's own HTTP-boundary JSON summary).
# path_regex override avoids Starlette's default Mount behavior of 307-
# redirecting the bare "/metrics" path to "/metrics/" before reaching the
# sub-app (same workaround vLLM's own instrumentator mounting code uses).
_metrics_route = Mount("/metrics", make_asgi_app())
_metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
app.routes.append(_metrics_route)
