"""Deterministic / batch-invariant mode helpers (Phase C — C3).

Wires ``VLLM_BATCH_INVARIANT`` on SM ≥ 8.0 and refuses otherwise. Application-
level warmup replaces the plan's ``VLLM_DETERMINISM_WARMUP_ITERATIONS`` name,
which does not exist in the pinned vLLM 0.22.1 (design.md D3).

``ensure_deterministic_mode()`` must run before the engine is constructed:
vLLM may capture CUDA graphs during construction, and a graph captured before
the batch-invariant dispatcher override is installed keeps replaying the
original (non-invariant) kernels no matter what the env var says afterward.
This is why only process startup calls this function for real — a per-request
call against an already-constructed engine cannot honor the guarantee, so
``ChatService`` refuses instead of calling it late (see
``services/chat_service.py::_enforce_deterministic``).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from inference_x.utils.cuda_env import (
    probe_compute_capability,
    supports_deterministic_batch_invariant,
)

if TYPE_CHECKING:
    from inference_x.engines.base import BaseEngine

logger = logging.getLogger(__name__)

_WARMUP_PROMPT = "ping"
_DEFAULT_WARMUP_ITERS = 3


class DeterminismUnsupportedError(ValueError):
    """Client demanded ``deterministic: true`` on unsupported hardware.

    Subclasses ``ValueError`` so ``api/errors.value_error_handler`` maps it to
    HTTP 400 without a new registration (same pattern as
    ``StrictModeViolationError``).
    """


def batch_invariant_env_enabled() -> bool:
    return os.environ.get("VLLM_BATCH_INVARIANT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def ensure_deterministic_mode(*, require: bool = True) -> bool:
    """Enable vLLM batch-invariant mode when the GPU supports it.

    Call this before the engine is constructed (see module docstring) — it is
    the process-startup activation path; per-request `deterministic: true` is
    enforced separately by refusing unless startup already activated this.

    Parameters
    ----------
    require:
        When True, raise ``DeterminismUnsupportedError`` if SM < 8.0 or CUDA
        is absent. When False, return False quietly on unsupported hosts.

    Returns
    -------
    bool
        True when batch-invariant mode is active after the call.
    """
    cap = probe_compute_capability()
    if not supports_deterministic_batch_invariant(cap):
        if require:
            sm = f"SM {cap[0]}.{cap[1]}" if cap else "no CUDA GPU"
            raise DeterminismUnsupportedError(
                f"deterministic: true requires compute capability SM ≥ 8.0 "
                f"(host reports {sm})"
            )
        return False

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm.model_executor.layers.batch_invariant import (  # type: ignore[import-untyped]
        init_batch_invariance,
    )

    # Let a genuine init failure propagate rather than reporting success the
    # manifest can't back up (refuse rather than lie, design.md D4).
    init_batch_invariance()

    logger.info(
        "Deterministic mode enabled (VLLM_BATCH_INVARIANT=1, capability=SM %s.%s)",
        cap[0] if cap else "?",
        cap[1] if cap else "?",
    )
    return True


async def warm_deterministic_engine(
    engine: BaseEngine, *, iterations: int = _DEFAULT_WARMUP_ITERS
) -> None:
    """Run N short greedy generates to settle batch-invariant kernels (D3)."""
    from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        model=getattr(engine, "model_name", "warmup"),
        messages=[ChatMessage(role="user", content=_WARMUP_PROMPT)],
        max_tokens=1,
        temperature=0.0,
        seed=0,
        deterministic=False,  # warmup itself must not re-enter refusal logic
    )
    for i in range(iterations):
        await engine.generate(request)
        logger.info("Deterministic warmup iteration %d/%d complete", i + 1, iterations)
