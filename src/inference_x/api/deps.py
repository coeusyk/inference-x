from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_x.core.settings import AppSettings, get_settings
from inference_x.engines.pool import EnginePool
from inference_x.engines.vllm_engine import VLLMEngine
from inference_x.observability.exporters import build_exporter
from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import InMemoryStorage
from inference_x.routing.admission import AdmissionController
from inference_x.routing.task_router import TaskRouter
from inference_x.routing.variant_selector import select_variant
from inference_x.services.chat_service import ChatService
from inference_x.services.metrics_service import MetricsService
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import (
    apply_tier_knobs,
    probe_gpu_memory_gib,
    validate_model_fits,
)
from inference_x.utils.vram_tiers import VramTier

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_registry(config_dir: str) -> ModelRegistry:
    return ModelRegistry.from_config(config_dir)


def _resolve_vram_tier_for_pool(config_dir: str) -> VramTier | None:
    """Resolve the VRAM tier for engine-knob wiring, fail-open with a warning.

    Same posture as _build_admission_controller: a resolution failure must not
    block engine construction, it only means apply_tier_knobs() leaves each
    model's config untouched (vLLM's own defaults, as before this knob wiring
    existed).
    """
    try:
        return get_settings().get_vram_tier()
    except Exception as exc:
        logger.warning(
            "VRAM tier resolution failed for engine knob wiring (using model/vLLM "
            "defaults): %s",
            exc,
        )
        return None


def _resolve_loaded_model_names(
    registry: ModelRegistry,
    loaded_models: tuple[str, ...],
    tier: VramTier | None,
    available_vram_gib: float,
) -> list[str]:
    """Resolve each name in *loaded_models* to a concrete registered model name.

    A name matching a registered ModelEntry.name exactly is used as-is
    (today's behavior, unchanged). Otherwise it's treated as a model family
    and resolved to its best-fitting variant via
    routing/variant_selector.select_variant() (see
    add-model-variant-routing) — requires a resolved *tier* to size variants
    against; if no tier is available, an unrecognized name falls through to
    registry.get()'s existing "not registered" error, unchanged from before
    this module existed.
    """
    resolved: list[str] = []
    for name in loaded_models:
        if name in registry:
            resolved.append(name)
            continue
        if tier is None or not registry.variants(name):
            registry.get(name)  # raises the standard "not registered" ValueError
        resolved.append(select_variant(name, registry, tier, available_vram_gib))
    return resolved


@lru_cache(maxsize=1)
def _build_engine_pool(config_dir: str, loaded_models: tuple[str, ...]) -> EnginePool:
    """Build an EnginePool with exactly one VLLMEngine for the configured model.

    B6 (Option A, docs/DECISIONS.md DEC-059): each server process serves
    exactly one model. Multi-model serving is achieved by running one
    process per model (see playground/server_control.py), not by loading
    multiple engines into one process — INFERENCE_X_LOADED_MODELS resolving
    to more than one distinct model is rejected here as a configuration
    error rather than silently loading a second engine into this process.
    """
    registry = _build_registry(config_dir)
    free_gib, total_gib = probe_gpu_memory_gib()
    total_vram = total_gib if total_gib is not None else 8.0
    available_vram = free_gib if free_gib is not None else total_vram
    tier = _resolve_vram_tier_for_pool(config_dir)
    resolved_models = _resolve_loaded_model_names(registry, loaded_models, tier, available_vram)

    distinct_models = list(dict.fromkeys(resolved_models))
    if not distinct_models:
        return EnginePool({})
    if len(distinct_models) > 1:
        raise ValueError(
            f"INFERENCE_X_LOADED_MODELS resolved to {len(distinct_models)} models "
            f"({', '.join(distinct_models)}), but each server process serves "
            "exactly one model (docs/DECISIONS.md DEC-059). Run one process per "
            "model instead — see playground/server_control.py for the pattern "
            "InferenceX's own TUI compare mode uses."
        )

    model_name = distinct_models[0]
    model_config = apply_tier_knobs(registry.get(model_name).model_dump(), tier)
    validate_model_fits(model_config, total_vram_gib=total_vram, free_vram_gib=free_gib)
    logger.info("Loading engine for model=%s", model_name)
    engine = VLLMEngine(model_config, free_vram_gib=free_gib, total_vram_gib=total_gib)
    return EnginePool({model_name: engine})


def _resolve_default_model(
    registry: ModelRegistry,
    default_model: str,
    tier: VramTier | None,
    available_vram_gib: float,
) -> str:
    """Resolve INFERENCE_X_DEFAULT_MODEL to a concrete registered model name.

    A concrete registered name passes through unchanged (existing behavior,
    zero change). A name matching a ModelEntry.family — and only when a VRAM
    tier is resolved — is resolved via variant_selector.select_variant().
    Closes the DEC-041 scope boundary: the router's default previously
    required an exact registered name.

    Any other value (not a concrete name, not a resolvable family, or no tier
    available to size variants against) is returned unchanged so
    TaskRouter's DefaultModelPolicy raises its own existing "not in the
    registry" error — preserving that error path exactly as it was before
    this function existed.
    """
    if default_model in registry or tier is None or not registry.variants(default_model):
        return default_model
    resolved = select_variant(default_model, registry, tier, available_vram_gib)
    logger.info(
        "Default model resolved: %s → %s (tier: %s)", default_model, resolved, tier.name
    )
    return resolved


@lru_cache(maxsize=1)
def _build_router(config_dir: str, default_model: str) -> TaskRouter:
    registry = _build_registry(config_dir)
    tier = _resolve_vram_tier_for_pool(config_dir)
    session_free_gib, session_total_gib = probe_gpu_memory_gib()
    available_vram = session_free_gib if session_free_gib is not None else (session_total_gib or 8.0)
    resolved_default = _resolve_default_model(registry, default_model, tier, available_vram)
    return TaskRouter(registry, resolved_default)


@lru_cache(maxsize=1)
def _build_admission_controller(config_dir: str) -> AdmissionController:
    """Build the AdmissionController with the resolved VRAM tier as its context cap.

    Falls back to AdmissionController's built-in default cap (no tier) if tier
    resolution fails — same fail-open-with-a-warning posture as the tier
    logging in initialize_app().
    """
    registry = _build_registry(config_dir)
    tier = None
    try:
        tier = get_settings().get_vram_tier()
    except Exception as exc:
        logger.warning(
            "VRAM tier resolution failed for admission control (using default cap): %s",
            exc,
        )
    settings = get_settings()
    return AdmissionController(
        registry,
        tier=tier,
        admission_wait_s=settings.admission_wait_s,
        batch_admission_wait_s=settings.batch_admission_wait_s,
        batch_waiter_multiplier=settings.batch_waiter_multiplier,
    )


def get_registry(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ModelRegistry:
    return _build_registry(settings.config_dir)


def get_vram_tier(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> VramTier | None:
    """VRAM tier for read-only reporting routes (`plan`/`doctor`) — same fail-open
    resolution engine construction uses (`_resolve_vram_tier_for_pool`)."""
    return _resolve_vram_tier_for_pool(settings.config_dir)


def get_chat_service(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> ChatService:
    pool = _build_engine_pool(settings.config_dir, tuple(settings.loaded_models))
    registry = _build_registry(settings.config_dir)
    router = _build_router(settings.config_dir, settings.default_model)
    admission = _build_admission_controller(settings.config_dir)
    return ChatService(engine_pool=pool, registry=registry, router=router, admission=admission)


def get_engine_pool(
    settings: Annotated[AppSettings, Depends(get_settings)],
) -> EnginePool:
    return _build_engine_pool(settings.config_dir, tuple(settings.loaded_models))


@lru_cache(maxsize=1)
def _build_recorder() -> MetricsRecorder:
    storage = InMemoryStorage()
    exporter = build_exporter()
    return MetricsRecorder(storage=storage, exporter=exporter)


def get_recorder() -> MetricsRecorder:
    """Return the process-level recorder singleton (not a FastAPI Depends)."""
    return _build_recorder()


def get_metrics_service() -> MetricsService:
    return MetricsService(_build_recorder())


def initialize_app() -> None:
    """Eagerly build registry, router, and engine pool at application startup.

    Raises on config or model-load failure so the process fails fast instead
    of accepting requests that would fail on first dependency resolution.
    """
    settings = get_settings()
    config_dir = settings.config_dir
    default_model = settings.default_model
    loaded_models = settings.loaded_models

    logger.info(
        "Initializing InferenceX (config_dir=%s, default_model=%s, loaded_models=%s)",
        config_dir,
        default_model,
        loaded_models,
    )

    registry = _build_registry(config_dir)
    logger.info("Model registry loaded: %s", registry.names())

    try:
        tier = settings.get_vram_tier()
        logger.info(
            "VRAM tier: %s (util_ceiling=%.2f, max_model_len_cap=%d, max_num_seqs=%d) — %s",
            tier.name,
            tier.gpu_memory_utilization_ceiling,
            tier.max_model_len_cap,
            tier.max_num_seqs,
            tier.description,
        )
    except Exception as exc:
        logger.warning("VRAM tier resolution failed (continuing without it): %s", exc)

    _build_router(config_dir, default_model)
    logger.info("Task router ready (default_model=%s)", default_model)

    pool = _build_engine_pool(config_dir, tuple(loaded_models))
    if not pool.all_healthy():
        unhealthy = [n for n, s in pool.health_status().items() if s != "ok"]
        raise RuntimeError(
            f"Engines not healthy after startup init: {unhealthy}"
        )

    logger.info("Engine pool ready: %s", pool.loaded_models())


def shutdown_app() -> None:
    """Release vLLM subprocesses and clear startup singleton caches."""
    settings = get_settings()
    if _build_engine_pool.cache_info().currsize:
        try:
            pool = _build_engine_pool(settings.config_dir, tuple(settings.loaded_models))
            pool.shutdown()
        except Exception as exc:
            logger.warning("Engine pool shutdown failed: %s", exc)
    _build_engine_pool.cache_clear()
    _build_registry.cache_clear()
    _build_router.cache_clear()
    _build_admission_controller.cache_clear()
    logger.info("InferenceX shutdown complete")
