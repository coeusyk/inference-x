"""GPU memory sizing for vLLM engines from model config and probed VRAM."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from inference_x.utils.vram_tiers import VramTier

logger = logging.getLogger(__name__)

_POOL_GPU_HEADROOM = 0.92
_BYTES_PER_PARAM = 2  # bf16/fp16 weights (vLLM dtype=auto typical) — default/unquantized
_RUNTIME_HEADROOM_GIB = 0.35  # fragmentation, activations, graph capture slack
_CUDAGRAPH_OVERHEAD_GIB = 0.45  # vLLM 0.21+ CUDA graph memory profiling reserve
_FREE_VRAM_SAFETY = 0.98
_DEFAULT_WEIGHT_GIB = 1.5
_DEFAULT_KV_GIB = 0.4

# Approximate on-GPU bytes/param by quantization scheme, matched against
# ModelEntry.quantization (vLLM's own quant method string, lowercased).
# 4-bit schemes include a small overhead for group-wise scales/zero-points,
# so they land a bit above a bare 0.5 bytes/param.
_QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "awq": 0.55,
    "awq_marlin": 0.55,
    "gptq": 0.55,
    "gptq_marlin": 0.55,
    "int4": 0.55,
    "bitsandbytes": 0.55,  # nf4/int4 default; int8 bnb configs are rarer, treat as 4-bit
    "int8": 1.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
}


def _bytes_per_param(quantization: str | None) -> float:
    """Resolve GPU bytes/param for a quantization scheme (None/unknown → bf16)."""
    if not quantization:
        return _BYTES_PER_PARAM
    key = str(quantization).strip().lower()
    if key in _QUANT_BYTES_PER_PARAM:
        return _QUANT_BYTES_PER_PARAM[key]
    # Substring fallback for variant spellings (e.g. "gptq-4bit", "marlin-awq").
    for name, bytes_per_param in _QUANT_BYTES_PER_PARAM.items():
        if name in key:
            return bytes_per_param
    logger.debug("Unknown quantization %r; assuming bf16 (2 bytes/param)", quantization)
    return _BYTES_PER_PARAM


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


def _embedding_param_count(config: dict[str, Any] | None, total_params: int) -> int:
    """Params in the embedding + (untied) lm_head layers.

    AWQ/GPTQ-style weight-only quantization calibrates the matmul weights in
    attention/MLP blocks; embedding lookups (and, when ``tie_word_embeddings``
    is false, the separate lm_head projection) are left at full precision.
    Uniformly applying the quant bytes/param ratio to every parameter misses
    this — see ``estimate_weight_gib``.
    """
    if not config:
        return 0
    hidden = int(config.get("hidden_size") or config.get("n_embd") or 0)
    vocab = int(config.get("vocab_size") or 0)
    if not hidden or not vocab:
        return 0
    multiplier = 1 if config.get("tie_word_embeddings", True) else 2
    return min(vocab * hidden * multiplier, total_params)


def estimate_weight_gib(model_path: str, quantization: str | None = None) -> float:
    """Estimate model weight VRAM from HuggingFace config (GiB).

    *quantization* is the ModelEntry.quantization value (e.g. ``"awq"``,
    ``"gptq"``, ``"int8"``); ``None`` assumes unquantized bf16/fp16 weights.

    Quantized estimates split out embedding/lm_head params and price them at
    bf16 instead of the quant ratio (see ``_embedding_param_count``) — without
    this split, qwen2.5-7b-awq's estimate (untied embeddings, 152k vocab)
    undercounted weight VRAM by ~2 GiB and the model failed to load in
    practice (2026-07-02).
    """
    config = _hf_config_dict(model_path)
    params = _params_from_name_or_path(model_path)
    if params is None and config:
        params = config.get("num_parameters") or config.get("n_params")
        if not params:
            params = _parameter_count_from_arch(config)
    if not params:
        return _DEFAULT_WEIGHT_GIB
    params = int(params)

    bytes_per_param = _bytes_per_param(quantization)
    if bytes_per_param >= _BYTES_PER_PARAM:
        return (params * bytes_per_param) / (1024**3)

    embed_params = _embedding_param_count(config, params)
    quantized_params = params - embed_params
    total_bytes = quantized_params * bytes_per_param + embed_params * _BYTES_PER_PARAM
    return total_bytes / (1024**3)


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


# Extra VRAM margin for architectures with measured overhead the generic
# weights+KV+runtime estimate doesn't model, keyed by a substring of the
# model_path/repo id. minicpm5-1b measured 6.55 GiB peak VRAM against a 3.79
# GiB estimated budget (2026-07-02) with no confirmed root cause (possibly a
# hybrid/non-standard component vLLM's own profiler doesn't see) — this is a
# conservative correction to avoid a repeat under-budget failure, not a
# diagnosed fix. Matched on model_path rather than HF config `model_type`:
# openbmb/MiniCPM5-1B's config self-reports architectures=["LlamaForCausalLM"]
# / model_type="llama" for tooling compatibility, so model_type can't
# distinguish it from a real Llama model.
_ARCH_OVERHEAD_GIB: dict[str, float] = {
    "minicpm": 2.8,
}


def _architecture_overhead_gib(model_path: str) -> float:
    """Conservative extra-VRAM margin for architectures with known-unmodeled overhead."""
    path_lower = model_path.lower()
    for name, overhead in _ARCH_OVERHEAD_GIB.items():
        if name in path_lower:
            logger.warning(
                "Model %s has measured VRAM overhead the generic estimator doesn't "
                "model; applying a conservative +%.1f GiB margin.",
                model_path, overhead,
            )
            return overhead
    return 0.0


def estimate_engine_footprint_gib(
    model_path: str, max_model_len: int, quantization: str | None = None
) -> float:
    """Total GiB budget for weights + KV cache + vLLM runtime/CUDA-graph overhead."""
    return (
        estimate_weight_gib(model_path, quantization)
        + estimate_kv_cache_gib(model_path, max_model_len)
        + _RUNTIME_HEADROOM_GIB
        + _CUDAGRAPH_OVERHEAD_GIB
        + _architecture_overhead_gib(model_path)
    )


def _minimum_utilization(
    model_path: str,
    max_model_len: int,
    total_vram_gib: float,
    quantization: str | None = None,
) -> float:
    """Lowest gpu_memory_utilization that fits this model on *total_vram_gib*."""
    if total_vram_gib <= 0:
        return 0.9
    return estimate_engine_footprint_gib(model_path, max_model_len, quantization) / total_vram_gib


def _weights_only_utilization(
    model_path: str, total_vram_gib: float, quantization: str | None = None
) -> float:
    """Lowest utilization that fits model weights alone (sequential pool fallback)."""
    if total_vram_gib <= 0:
        return 0.5
    return estimate_weight_gib(model_path, quantization) / total_vram_gib


def _multi_engine_overhead_gib(total_vram_gib: float) -> float:
    """VRAM held outside utilization fractions; scales with GPU size."""
    return min(3.35, total_vram_gib * 0.42)


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
    quantization = config.get("quantization")
    user_cap = _user_util_cap(config)

    equal_share = _POOL_GPU_HEADROOM / pool_size
    floor = _minimum_utilization(model_path, max_model_len, total_vram_gib, quantization)

    weights = [
        estimate_weight_gib(str(c.get("model_path", "")), c.get("quantization"))
        for c in pool_configs
    ]
    total_weight = sum(weights) or 1.0
    my_weight = estimate_weight_gib(model_path, quantization)
    weight_share = my_weight / total_weight

    mem_budget_gib = estimate_engine_footprint_gib(model_path, max_model_len, quantization) * (
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
    weights_floor = _weights_only_utilization(model_path, total_vram_gib, config.get("quantization"))
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
                c.get("quantization"),
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
    quantization = config.get("quantization")
    user_cap = _user_util_cap(config)
    weights_floor = _weights_only_utilization(model_path, total_vram_gib, quantization)
    footprint_util = _minimum_utilization(model_path, max_model_len, total_vram_gib, quantization)

    max_from_free: float | None = None
    if free_vram_gib is not None and total_vram_gib > 0:
        max_from_free = (_FREE_VRAM_SAFETY * free_vram_gib) / total_vram_gib
        weight_gib = estimate_weight_gib(model_path, quantization)
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
            c.get("quantization"),
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
            c.get("quantization"),
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


def apply_tier_knobs(config: dict[str, Any], tier: VramTier | None) -> dict[str, Any]:
    """Return a copy of *config* with VRAM-tier engine knobs resolved and merged in.

    ``max_num_seqs`` / ``max_num_batched_tokens``: the tighter of any per-model
    override already present in *config* and the tier's ceiling — a model can
    only ask for a smaller budget than its tier allows, never a looser one
    (same composition ``AdmissionController._context_ceiling`` uses for
    ``max_model_len``).

    ``block_size`` / ``kv_cache_dtype`` / ``enable_prefix_caching``: tier-only,
    no per-model override — ``ModelEntry`` declares none of these.

    A *tier* of ``None`` (VRAM tier resolution failed) leaves *config*
    unchanged, consistent with this codebase's fail-open posture for advisory
    signals elsewhere (e.g. VRAM tier resolution failure in ``api/deps.py``).
    """
    resolved = dict(config)
    if tier is None:
        return resolved

    model_max_num_seqs = resolved.get("max_num_seqs")
    resolved["max_num_seqs"] = (
        min(model_max_num_seqs, tier.max_num_seqs)
        if model_max_num_seqs is not None
        else tier.max_num_seqs
    )

    if tier.max_num_batched_tokens is not None:
        model_max_batched = resolved.get("max_num_batched_tokens")
        resolved["max_num_batched_tokens"] = (
            min(model_max_batched, tier.max_num_batched_tokens)
            if model_max_batched is not None
            else tier.max_num_batched_tokens
        )

    resolved["block_size"] = tier.block_size
    resolved["kv_cache_dtype"] = tier.kv_cache_dtype
    resolved["enable_prefix_caching"] = tier.enable_prefix_caching
    return resolved


def scale_model_config_for_pool(
    config: dict[str, Any],
    *,
    pool_size: int,
    pool_models: list[str] | None = None,
    pool_configs: list[dict[str, Any]] | None = None,
    total_vram_gib: float = 8.0,
    engine_index: int = 0,
    free_vram_gib: float | None = None,
    session_free_vram_gib: float | None = None,
) -> dict[str, Any]:
    """Return a copy of *config* with gpu_memory_utilization sized for this GPU.

    ``gpu_memory_utilization: "auto"`` and an explicit float both flow through
    the same footprint-aware sizing below. "auto" means "no user-set ceiling"
    (see _user_util_cap()) — it does NOT mean "ignore the model's own weight
    and KV-cache footprint and grab a flat fraction of free VRAM regardless of
    model size." A prior version special-cased "auto" to do exactly that,
    which is why a 125M-param model could claim >85% of an 8 GiB GPU: fixed.
    """
    scaled = dict(config)
    if pool_configs is None:
        pool_configs = [scaled]
    is_auto = scaled.get("gpu_memory_utilization") == "auto"

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
        "GPU memory for model=%s: utilization=%.3f%s (total=%.1f GiB, free=%s)",
        scaled.get("name"),
        util,
        " [auto, footprint-sized]" if is_auto else "",
        total_vram_gib,
        f"{free_vram_gib:.1f} GiB" if free_vram_gib is not None else "unknown",
    )
    return scaled
