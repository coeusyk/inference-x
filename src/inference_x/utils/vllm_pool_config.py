"""GPU memory sizing for vLLM engines from model config and probed VRAM."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from inference_x.benchmarks.hardware import suggest_gpu_memory_utilization

logger = logging.getLogger(__name__)

_POOL_GPU_HEADROOM = 0.92
_BYTES_PER_PARAM = 2  # bf16/fp16 weights (vLLM dtype=auto typical)
_RUNTIME_HEADROOM_GIB = 0.35  # fragmentation, activations, graph capture slack
_CUDAGRAPH_OVERHEAD_GIB = 0.45  # vLLM 0.21+ CUDA graph memory profiling reserve
_FREE_VRAM_SAFETY = 0.98
_DEFAULT_WEIGHT_GIB = 1.5
_DEFAULT_KV_GIB = 0.4


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


@lru_cache(maxsize=32)
def _hf_config_dict(model_path: str) -> dict[str, Any] | None:
    """Load HuggingFace model config metadata (cached). Returns None on failure."""
    if not model_path or model_path.startswith(("/", "./", "../")):
        return None
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path, trust_remote_code=True).to_dict()
    except Exception as exc:
        logger.debug("HF config unavailable for %s: %s", model_path, exc)
        return None


def _parameter_count_from_arch(config: dict[str, Any]) -> int:
    """Rough parameter count for decoder-only transformers when num_parameters absent."""
    hidden = int(config.get("hidden_size") or config.get("n_embd") or 4096)
    layers = int(
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32
    )
    vocab = int(config.get("vocab_size") or 32000)
    intermediate = int(config.get("intermediate_size") or hidden * 4)
    # Embedding + per-layer attention + MLP (order-of-magnitude).
    per_layer = 12 * hidden * hidden + 13 * hidden * intermediate
    return layers * per_layer + vocab * hidden


def _params_from_name_or_path(model_path: str) -> int | None:
    """Parse parameter count hints from repo ids like ``Qwen2.5-0.5B`` or ``opt-125m``."""
    match = re.search(r"(\d+(?:\.\d+)?)([mMbB])\b", model_path)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def estimate_weight_gib(model_path: str) -> float:
    """Estimate model weight VRAM from HuggingFace config (GiB)."""
    config = _hf_config_dict(model_path)
    params = _params_from_name_or_path(model_path)
    if params is None and config:
        params = config.get("num_parameters") or config.get("n_params")
        if not params:
            params = _parameter_count_from_arch(config)
    if not params:
        return _DEFAULT_WEIGHT_GIB
    return (int(params) * _BYTES_PER_PARAM) / (1024**3)


def estimate_kv_cache_gib(model_path: str, max_model_len: int) -> float:
    """Estimate KV-cache VRAM from architecture and max_model_len (GiB)."""
    config = _hf_config_dict(model_path)
    if not config:
        return _DEFAULT_KV_GIB * (max_model_len / 2048)
    hidden = int(config.get("hidden_size") or config.get("n_embd") or 4096)
    layers = int(
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or 32
    )
    # keys + values, per token, all layers.
    bytes_needed = 2 * layers * hidden * max_model_len * _BYTES_PER_PARAM
    return bytes_needed / (1024**3)


def estimate_engine_footprint_gib(model_path: str, max_model_len: int) -> float:
    """Total GiB budget for weights + KV cache + vLLM runtime/CUDA-graph overhead."""
    return (
        estimate_weight_gib(model_path)
        + estimate_kv_cache_gib(model_path, max_model_len)
        + _RUNTIME_HEADROOM_GIB
        + _CUDAGRAPH_OVERHEAD_GIB
    )


def _minimum_utilization(model_path: str, max_model_len: int, total_vram_gib: float) -> float:
    """Lowest gpu_memory_utilization that fits this model on *total_vram_gib*."""
    if total_vram_gib <= 0:
        return 0.9
    return estimate_engine_footprint_gib(model_path, max_model_len) / total_vram_gib


def _weights_only_utilization(model_path: str, total_vram_gib: float) -> float:
    """Lowest utilization that fits model weights alone (sequential pool fallback)."""
    if total_vram_gib <= 0:
        return 0.5
    return estimate_weight_gib(model_path) / total_vram_gib


def _multi_engine_overhead_gib(total_vram_gib: float) -> float:
    """VRAM held outside utilization fractions; scales with GPU size."""
    return min(3.35, total_vram_gib * 0.42)


def resolve_gpu_memory_utilization(
    value: str | float,
    *,
    model_name: str = "",
    max_model_len: int | None = None,
) -> float:
    """Resolve models.yaml gpu_memory_utilization, including the ``auto`` sentinel."""
    if value == "auto":
        buffer = float(os.getenv("INFERENCEX_VRAM_SAFETY_BUFFER_GB", "0.4"))
        from inference_x.benchmarks.hardware import _vram_for_utilization

        free_gib, total_gib, source = _vram_for_utilization()
        resolved = suggest_gpu_memory_utilization()
        max_len_label = (
            str(max_model_len)
            if max_model_len is not None
            else "vLLM default (uncapped)"
        )
        if source != "none":
            logger.info(
                "Loading %s\n"
                "  gpu_memory_utilization: auto → %.2f\n"
                "    (%.2f GB free − %.2f GB buffer) / %.2f GB total [%s]\n"
                "  max_model_len: %s",
                model_name or "?",
                resolved,
                free_gib,
                buffer,
                total_gib,
                source,
                max_len_label,
            )
        else:
            logger.info(
                "Loading %s\n"
                "  gpu_memory_utilization: auto → %.2f (no GPU detected)\n"
                "  max_model_len: %s",
                model_name or "?",
                resolved,
                max_len_label,
            )
        return resolved
    return float(value)


def _user_util_cap(config: dict[str, Any]) -> float | None:
    """Optional user ceiling from models.yaml (omit to fully auto-size)."""
    raw = config.get("gpu_memory_utilization")
    if raw is None or raw == "auto":
        return None
    return float(raw)


def _weight_scaled_utilization(
    config: dict[str, Any],
    *,
    pool_size: int,
    pool_configs: list[dict[str, Any]],
    total_vram_gib: float,
) -> float:
    """Weight-aware utilization before sequential VRAM caps."""
    name = str(config.get("name", ""))
    model_path = str(config.get("model_path", ""))
    max_model_len = int(config.get("max_model_len") or 2048)
    user_cap = _user_util_cap(config)

    equal_share = _POOL_GPU_HEADROOM / pool_size
    floor = _minimum_utilization(model_path, max_model_len, total_vram_gib)

    weights = [estimate_weight_gib(str(c.get("model_path", ""))) for c in pool_configs]
    total_weight = sum(weights) or 1.0
    my_weight = estimate_weight_gib(model_path)
    weight_share = my_weight / total_weight

    mem_budget_gib = estimate_engine_footprint_gib(model_path, max_model_len) * (
        0.85 + 0.15 * weight_share
    )
    weight_based = mem_budget_gib / total_vram_gib

    candidates = [equal_share, weight_based, floor]
    if user_cap is not None:
        candidates.append(user_cap)
    effective = min(candidates)
    effective = max(effective, floor)
    return round(min(effective, _POOL_GPU_HEADROOM), 4)


def _apply_sequential_vram_caps(
    util: float,
    config: dict[str, Any],
    *,
    engine_index: int,
    pool_size: int,
    pool_configs: list[dict[str, Any]],
    total_vram_gib: float,
    free_vram_gib: float | None,
) -> float:
    """Cap utilization so vLLM's free-memory check passes for sequential pool loads."""
    model_path = str(config.get("model_path", ""))
    weights_floor = _weights_only_utilization(model_path, total_vram_gib)
    capped = util

    if free_vram_gib is not None and total_vram_gib > 0:
        max_from_free = (_FREE_VRAM_SAFETY * free_vram_gib) / total_vram_gib
        capped = min(capped, max_from_free)

    if engine_index < pool_size - 1:
        remaining = pool_configs[engine_index + 1 :]
        max_next_util = max(
            _minimum_utilization(
                str(c.get("model_path", "")),
                int(c.get("max_model_len") or 2048),
                total_vram_gib,
            )
            for c in remaining
        )
        overhead = _multi_engine_overhead_gib(total_vram_gib)
        max_current = 1.0 - (overhead / total_vram_gib) - max_next_util
        capped = min(capped, max_current)

    capped = round(capped, 4)
    if capped + 1e-6 < weights_floor:
        raise ValueError(
            f"Model {config.get('name')} cannot fit in the remaining GPU memory for this pool. "
            f"Need gpu_memory_utilization >= {weights_floor:.3f} but capped at {capped:.3f}. "
            "Load fewer models or use a GPU with more VRAM."
        )
    return capped


def _single_engine_utilization(
    config: dict[str, Any],
    *,
    total_vram_gib: float,
    free_vram_gib: float | None,
) -> float:
    """Compute gpu_memory_utilization for a lone engine from footprint + free VRAM."""
    model_path = str(config.get("model_path", ""))
    max_model_len = int(config.get("max_model_len") or 2048)
    user_cap = _user_util_cap(config)
    weights_floor = _weights_only_utilization(model_path, total_vram_gib)
    footprint_util = _minimum_utilization(model_path, max_model_len, total_vram_gib)

    max_from_free: float | None = None
    if free_vram_gib is not None and total_vram_gib > 0:
        max_from_free = (_FREE_VRAM_SAFETY * free_vram_gib) / total_vram_gib
        weight_gib = estimate_weight_gib(model_path)
        if weight_gib > free_vram_gib * _FREE_VRAM_SAFETY:
            raise ValueError(
                f"Only {free_vram_gib:.1f} GiB VRAM free but model {config.get('name')} "
                f"needs ~{weight_gib:.1f} GiB for weights. Stop other GPU processes "
                "(orphaned VLLM::EngineCore from prior runs) or choose a smaller model."
            )

    util = footprint_util
    if max_from_free is not None:
        util = min(util, max_from_free)
    if user_cap is not None:
        util = min(util, user_cap)
    util = max(util, weights_floor)
    if max_from_free is not None:
        util = min(util, max_from_free)
    return round(min(util, _POOL_GPU_HEADROOM), 4)


def validate_pool_fits(
    model_configs: list[dict[str, Any]],
    *,
    total_vram_gib: float = 8.0,
) -> None:
    """Raise ValueError when the pool cannot fit on probed VRAM."""
    if len(model_configs) <= 1:
        return

    needed_gib = sum(
        estimate_engine_footprint_gib(
            str(c.get("model_path", "")),
            int(c.get("max_model_len") or 2048),
        )
        for c in model_configs
    )
    budget_gib = total_vram_gib * _POOL_GPU_HEADROOM
    if needed_gib > budget_gib:
        names = ", ".join(str(c.get("name", "?")) for c in model_configs)
        raise ValueError(
            f"Models [{names}] need ~{needed_gib:.1f} GiB on GPU but only "
            f"~{budget_gib:.1f} GiB is available ({total_vram_gib:.1f} GiB total × "
            f"{_POOL_GPU_HEADROOM:.0%} headroom). Load fewer models, use smaller "
            "max_model_len values, or run separate server instances."
        )

    needed = sum(
        _minimum_utilization(
            str(c.get("model_path", "")),
            int(c.get("max_model_len") or 2048),
            total_vram_gib,
        )
        for c in model_configs
    )
    if needed > _POOL_GPU_HEADROOM:
        names = ", ".join(str(c.get("name", "?")) for c in model_configs)
        raise ValueError(
            f"Models [{names}] need ~{needed:.0%} of GPU memory combined on a "
            f"{total_vram_gib:.0f} GiB GPU (cap {_POOL_GPU_HEADROOM:.0%}). "
            "Load fewer models, use smaller models, or run separate server instances."
        )

    pool_size = len(model_configs)
    for idx, config in enumerate(model_configs):
        util = _weight_scaled_utilization(
            config,
            pool_size=pool_size,
            pool_configs=model_configs,
            total_vram_gib=total_vram_gib,
        )
        try:
            _apply_sequential_vram_caps(
                util,
                config,
                engine_index=idx,
                pool_size=pool_size,
                pool_configs=model_configs,
                total_vram_gib=total_vram_gib,
                free_vram_gib=None,
            )
        except ValueError as exc:
            names = ", ".join(str(c.get("name", "?")) for c in model_configs)
            raise ValueError(
                f"Models [{names}] cannot load sequentially on a {total_vram_gib:.0f} GiB "
                f"GPU: {exc}"
            ) from exc


def scale_model_config_for_pool(
    config: dict[str, Any],
    *,
    pool_size: int,
    pool_models: list[str] | None = None,
    pool_configs: list[dict[str, Any]] | None = None,
    total_vram_gib: float = 8.0,
    engine_index: int = 0,
    free_vram_gib: float | None = None,
) -> dict[str, Any]:
    """Return a copy of *config* with gpu_memory_utilization sized for this GPU."""
    scaled = dict(config)
    if pool_configs is None:
        pool_configs = [scaled]

    if scaled.get("gpu_memory_utilization") == "auto":
        max_model_len = scaled.get("max_model_len")
        util = resolve_gpu_memory_utilization(
            "auto",
            model_name=str(scaled.get("name", "")),
            max_model_len=int(max_model_len) if max_model_len is not None else None,
        )
        scaled["gpu_memory_utilization"] = util
        return scaled

    if pool_size <= 1:
        util = _single_engine_utilization(
            scaled,
            total_vram_gib=total_vram_gib,
            free_vram_gib=free_vram_gib,
        )
    else:
        util = _weight_scaled_utilization(
            scaled,
            pool_size=pool_size,
            pool_configs=pool_configs,
            total_vram_gib=total_vram_gib,
        )
        util = _apply_sequential_vram_caps(
            util,
            scaled,
            engine_index=engine_index,
            pool_size=pool_size,
            pool_configs=pool_configs,
            total_vram_gib=total_vram_gib,
            free_vram_gib=free_vram_gib,
        )
        if "max_model_len" in scaled:
            scaled["max_model_len"] = min(int(scaled["max_model_len"]), 2048)

    scaled["gpu_memory_utilization"] = util
    logger.info(
        "GPU memory for model=%s: utilization=%.3f (total=%.1f GiB, free=%s)",
        scaled.get("name"),
        util,
        total_vram_gib,
        f"{free_vram_gib:.1f} GiB" if free_vram_gib is not None else "unknown",
    )
    return scaled
