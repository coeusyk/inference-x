"""Helpers for sizing vLLM engines when multiple models share one GPU."""

from __future__ import annotations

from typing import Any

_POOL_GPU_HEADROOM = 0.92

_WEIGHT_GIB_ESTIMATES: dict[str, float] = {
    "opt-125m": 0.25,
    "qwen2.5-0.5b": 1.0,
    "tinyllama-chat": 2.05,
    "qwen2.5-1.5b": 3.0,
    "llama3-8b": 16.0,
}

_DEFAULT_WEIGHT_GIB = 1.5
# Weight + KV cache + CUDA graph capture headroom per engine (GiB).
_KV_GRAPH_HEADROOM_GIB = 1.0
# Non-utilization GPU memory retained after engine init (calibrated on 8 GiB WSL2).
_MULTI_ENGINE_OVERHEAD_GIB = 3.35
_FREE_VRAM_SAFETY = 0.98


def probe_gpu_memory_gib() -> tuple[float | None, float | None]:
    """Return (free_gib, total_gib) for cuda:0, or (None, None) if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024**3), total / (1024**3)
    except Exception:
        pass
    return None, None


def _estimate_weight_gib(model_name: str) -> float:
    return _WEIGHT_GIB_ESTIMATES.get(model_name, _DEFAULT_WEIGHT_GIB)


def _minimum_utilization(model_name: str, total_vram_gib: float) -> float:
    """Lowest gpu_memory_utilization that fits weights + minimal KV/graph memory."""
    return (_estimate_weight_gib(model_name) + _KV_GRAPH_HEADROOM_GIB) / total_vram_gib


def _weights_only_utilization(model_name: str, total_vram_gib: float) -> float:
    """Lowest utilization that fits model weights alone (sequential pool fallback)."""
    return _estimate_weight_gib(model_name) / total_vram_gib


def _weight_scaled_utilization(
    name: str,
    configured: float,
    *,
    pool_size: int,
    pool_models: list[str],
    total_vram_gib: float,
) -> float:
    """Weight-aware utilization before sequential VRAM caps."""
    equal_share = _POOL_GPU_HEADROOM / pool_size
    floor = _minimum_utilization(name, total_vram_gib)
    weights = [_estimate_weight_gib(m) for m in pool_models]
    total_weight = sum(weights)
    my_weight = _estimate_weight_gib(name)
    weight_share = my_weight / total_weight if total_weight else 1.0
    mem_budget_gib = my_weight + _KV_GRAPH_HEADROOM_GIB * (1.0 + weight_share)
    weight_based = mem_budget_gib / total_vram_gib
    effective = min(configured, equal_share, weight_based)
    effective = max(effective, floor)
    return round(min(effective, _POOL_GPU_HEADROOM), 4)


def _apply_sequential_vram_caps(
    util: float,
    name: str,
    *,
    engine_index: int,
    pool_size: int,
    pool_models: list[str],
    total_vram_gib: float,
    free_vram_gib: float | None,
) -> float:
    """Cap utilization so vLLM's free-memory check passes for sequential pool loads."""
    weights_floor = _weights_only_utilization(name, total_vram_gib)
    capped = util

    if free_vram_gib is not None and total_vram_gib > 0:
        max_from_free = (_FREE_VRAM_SAFETY * free_vram_gib) / total_vram_gib
        capped = min(capped, max_from_free)

    if engine_index < pool_size - 1:
        remaining = pool_models[engine_index + 1 :]
        max_next_util = max(
            _minimum_utilization(m, total_vram_gib) for m in remaining
        )
        max_current = (
            1.0 - (_MULTI_ENGINE_OVERHEAD_GIB / total_vram_gib) - max_next_util
        )
        capped = min(capped, max_current)

    capped = round(max(weights_floor, capped), 4)
    if capped + 1e-6 < weights_floor and (free_vram_gib is not None or engine_index > 0):
        raise ValueError(
            f"Model {name} cannot fit in the remaining GPU memory for this pool. "
            f"Need gpu_memory_utilization >= {weights_floor:.3f} but capped at {capped:.3f}. "
            "Load fewer models or use a GPU with more VRAM."
        )
    return capped


def validate_pool_fits(
    model_names: list[str],
    *,
    total_vram_gib: float = 8.0,
) -> None:
    """Raise ValueError when the combined minimum footprint exceeds VRAM."""
    if len(model_names) <= 1:
        return
    needed = sum(_minimum_utilization(n, total_vram_gib) for n in model_names)
    if needed > _POOL_GPU_HEADROOM:
        names = ", ".join(model_names)
        raise ValueError(
            f"Models [{names}] need ~{needed:.0%} of GPU memory combined on a "
            f"{total_vram_gib:.0f} GiB GPU (cap {_POOL_GPU_HEADROOM:.0%}). "
            "Load fewer models, use smaller models, or run separate server instances."
        )


def scale_model_config_for_pool(
    config: dict[str, Any],
    *,
    pool_size: int,
    pool_models: list[str] | None = None,
    total_vram_gib: float = 8.0,
    engine_index: int = 0,
    free_vram_gib: float | None = None,
) -> dict[str, Any]:
    """Return a copy of *config* sized for multi-engine sharing on one GPU."""
    if pool_size <= 1:
        return config

    scaled = dict(config)
    name = str(config.get("name", ""))
    configured = float(scaled.get("gpu_memory_utilization", 0.9))

    util = _weight_scaled_utilization(
        name,
        configured,
        pool_size=pool_size,
        pool_models=pool_models or [name],
        total_vram_gib=total_vram_gib,
    )
    util = _apply_sequential_vram_caps(
        util,
        name,
        engine_index=engine_index,
        pool_size=pool_size,
        pool_models=pool_models or [name],
        total_vram_gib=total_vram_gib,
        free_vram_gib=free_vram_gib,
    )
    scaled["gpu_memory_utilization"] = util

    if "max_model_len" in scaled:
        scaled["max_model_len"] = min(int(scaled["max_model_len"]), 2048)
    else:
        scaled["max_model_len"] = 2048

    return scaled
