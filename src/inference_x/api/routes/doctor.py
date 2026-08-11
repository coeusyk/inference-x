from typing import Annotated

from fastapi import APIRouter, Depends

from inference_x.api.deps import get_registry, get_vram_tier
from inference_x.benchmarks.hardware import extended_hardware_fields, profile_hardware
from inference_x.schemas.chat import ManifestHardware
from inference_x.schemas.diagnostics import DoctorModelFit, DoctorResponse
from inference_x.services.model_service import ModelRegistry
from inference_x.utils.vllm_pool_config import (
    apply_tier_knobs,
    probe_gpu_memory_gib,
    validate_model_fits,
)
from inference_x.utils.vram_tiers import VramTier

router = APIRouter()


def _hardware_snapshot() -> ManifestHardware:
    """Same two functions C1's manifest hardware block uses (design.md D4) — a fresh read
    each call rather than chat_service's process-lifetime cache: doctor is an occasional
    diagnostic call, not a per-request hot path, so a fresh probe costs nothing that matters.
    """
    hw_profile = profile_hardware()
    hw_extra = extended_hardware_fields()
    return ManifestHardware(
        gpu=hw_profile.gpu_name,
        vram_total_gib=hw_profile.vram_total_gb if hw_profile.has_gpu else None,
        driver=hw_extra.get("driver"),
        cuda=hw_extra.get("cuda"),
        cpu=hw_extra.get("cpu"),
        ram_gib=hw_profile.ram_total_gb,
        wsl2=hw_extra.get("wsl2"),
    )


@router.get(
    "/doctor", response_model=DoctorResponse, summary="Read-only environment and model-fit readiness"
)
def doctor(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    tier: Annotated[VramTier | None, Depends(get_vram_tier)],
) -> DoctorResponse:
    """Report probed GPU memory, the hardware profile, and per-model VRAM-fit readiness.
    Read-only: no engine is started, stopped, or reconfigured to answer this request.
    """
    free_gib, total_gib = probe_gpu_memory_gib()
    total_vram = total_gib if total_gib is not None else 8.0

    fits: list[DoctorModelFit] = []
    for model_entry in registry.all():
        config = apply_tier_knobs(model_entry.model_dump(), tier)
        try:
            validate_model_fits(
                config, total_vram_gib=total_vram, free_vram_gib=free_gib
            )
            fits.append(DoctorModelFit(model=model_entry.name, fits=True))
        except ValueError as exc:
            fits.append(
                DoctorModelFit(model=model_entry.name, fits=False, reason=str(exc))
            )

    return DoctorResponse(
        hardware=_hardware_snapshot(),
        vram_free_gib=free_gib,
        vram_total_gib=total_gib,
        models=fits,
    )
