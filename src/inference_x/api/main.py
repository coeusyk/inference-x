from fastapi import FastAPI
from .v1.completions import router as v1_router

app = FastAPI(title="LLM Serving Platform")

app.include_router(v1_router, prefix="/v1")

# Phase 3 will add: app.add_middleware(...)
# Phase 2 will modify: get_engine dependency only
