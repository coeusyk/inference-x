import logging

from fastapi import FastAPI

from inference_x.api.errors import runtime_error_handler, value_error_handler
from inference_x.api.routes.chat_completions import router as chat_router
from inference_x.api.routes.health import router as health_router
from inference_x.api.routes.models import router as models_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="InferenceX",
    description="Self-hosted vLLM-backed OpenAI-compatible inference API",
    version="0.1.0",
)

app.add_exception_handler(RuntimeError, runtime_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]

app.include_router(health_router)
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
