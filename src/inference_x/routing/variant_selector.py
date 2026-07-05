"""Load-time model variant selection: pick the best-fitting quantization variant.

A model "family" (ModelEntry.family) groups multiple config entries that are
the same logical model at different precisions (e.g. bf16/int8/awq of
qwen2.5-7b). select_variant() picks the highest-precision variant whose
estimated weight size fits the available VRAM budget — reusing
utils/vllm_pool_config's existing quant-aware bytes-per-param table as the
sole source of precision ordering, so there is exactly one place that knows
"awq is lower precision than bf16," not a second duplicated rank field that
could silently drift out of sync with it.

This is a load-time decision only: which concrete engine to start for a
family name given in INFERENCE_X_LOADED_MODELS. TaskRouter and
AdmissionController are unaffected — they only ever see the concrete,
already-resolved model name, exactly as before this module existed.
"""
from __future__ import annotations

from inference_x.schemas.model import ModelEntry
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import _bytes_per_param, estimate_weight_gib
from inference_x.utils.vram_tiers import VramTier


class NoVariantFitsError(RuntimeError):
    """No variant in a family fits the available VRAM budget.

    Raised at load time (before any engine is constructed) so the process
    fails fast instead of silently starting with no working model — matches
    this codebase's existing "engines not healthy after startup" fail-fast
    posture (see api/deps.initialize_app).
    """


def _sorted_by_precision(variants: list[ModelEntry]) -> list[ModelEntry]:
    """Highest precision (most bytes/param — bf16) first, 4-bit last."""
    return sorted(variants, key=lambda m: _bytes_per_param(m.quantization), reverse=True)


def select_variant(
    family: str,
    registry: ModelRegistry,
    tier: VramTier,
    available_vram_gib: float,
) -> str:
    """Return the name of the highest-precision variant of *family* that fits.

    "Fits" means estimate_weight_gib(variant) <= available_vram_gib *
    tier.gpu_memory_utilization_ceiling — the same headroom ceiling the rest
    of the VRAM-aware sizing path already uses.

    Raises:
        NoVariantFitsError: no variant in the family fits the budget, or the
            family name matches no registered entry at all.
    """
    variants = registry.variants(family)
    if not variants:
        raise NoVariantFitsError(
            f"No model variants registered for family '{family}'. "
            f"Available models: {registry.names()}"
        )

    budget_gib = available_vram_gib * tier.gpu_memory_utilization_ceiling
    ordered = _sorted_by_precision(variants)
    for variant in ordered:
        estimated_gib = estimate_weight_gib(variant.model_path, variant.quantization)
        if estimated_gib <= budget_gib:
            return variant.name

    sizes = ", ".join(
        f"{v.name} (~{estimate_weight_gib(v.model_path, v.quantization):.2f} GiB)"
        for v in ordered
    )
    raise NoVariantFitsError(
        f"No variant of family '{family}' fits the available VRAM budget "
        f"(~{budget_gib:.2f} GiB after the {tier.name} tier's "
        f"{tier.gpu_memory_utilization_ceiling:.0%} utilization ceiling). "
        f"Variants and estimated weight sizes: {sizes}."
    )
