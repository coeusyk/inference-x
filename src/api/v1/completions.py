from fastapi import APIRouter, Depends
from src.core.schemas import CompletionRequest
from src.core.engine import LLMEngine, VLLMEngine

router = APIRouter()

# Phase 1: Simple factory
def get_engine() -> LLMEngine:
    # Load from config/models.yaml
    return VLLMEngine(config)


@router.post("/chat/completions")
async def chat_completions(
    request: CompletionRequest,
    engine: LLMEngine = Depends(get_engine)
):
    return await engine.generate(request)


@router.get("/health")
async def health(engine: LLMEngine = Depends(get_engine)):
    return {"status": "healthy" if engine.health_check() else "unhealthy"}
