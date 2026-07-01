"""Pre-dispatch admission control: context-length and KV-budget enforcement.

AdmissionController runs after TaskRouter.select() and before the engine is
invoked. It never touches the GPU directly — it only decides whether a request
is admitted as-is, admitted with a clamped `max_tokens`, or rejected. All GPU
sizing lives in utils/vllm_pool_config.py and the engine, as in the rest of
routing/ (see routing/base.py).

Three independent gates, each keyed on real numbers where they're available:

1. Sequence-concurrency ceiling: the number of in-flight requests against a
   model must stay at or below the resolved max_num_seqs (tier ceiling ∩ any
   per-model ModelEntry.max_num_seqs override — see
   utils/vllm_pool_config.apply_tier_knobs for the same composition applied at
   engine-construction time). Unlike the two gates below, there is no clamp
   path: a request either gets a sequence slot or it doesn't, so both
   'interactive' and 'batch' priority get EngineSaturatedError when saturated.

2. Context length: prompt_tokens + requested_output_tokens must fit the
   model's context window (min of ModelEntry.max_model_len, the resolved VRAM
   tier's max_model_len_cap, and the request's own max_context_tokens, if
   set). Prompt tokens come from the engine's tokenizer when it exposes
   count_prompt_tokens(); otherwise a chars/4 heuristic is used.

3. KV-pool pressure: a per-model in-memory counter tracks tokens reserved by
   in-flight requests, compared against the engine's real post-load
   kv_capacity_tokens (num_gpu_blocks * block_size) when it exposes one.
   'interactive' requests get max_tokens clamped to fit; 'batch' requests are
   rejected with EngineSaturatedError (mapped to 429 + Retry-After) instead of
   silently getting a truncated completion.

All three gates degrade to "don't block" when the underlying number is
unavailable (no resolved tier, no tokenizer, no reported KV capacity) —
consistent with the rest of this codebase's fail-open-with-a-log posture for
advisory/best-effort signals (e.g. VRAM tier resolution in api/deps.py).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from inference_x.schemas.chat import ChatCompletionRequest
from inference_x.services.model_service import ModelRegistry

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_CAP = 4096  # matches the old ChatService._GLOBAL_MAX_TOKENS default
_DEFAULT_KV_SAFETY_MARGIN = 0.9  # leave headroom below the real KV pool size
_MIN_CLAMPED_OUTPUT_TOKENS = 16  # below this a "clamp" is really a rejection in disguise
_CHARS_PER_TOKEN_FALLBACK = 4  # used only when the engine has no tokenizer to ask


class ContextTooLongError(ValueError):
    """Prompt (+ requested output) exceeds the model's context window.

    Subclasses ValueError so it is handled by the existing sanitized 400
    handler (api/errors.value_error_handler) without a new registration.
    """


class EngineSaturatedError(Exception):
    """KV pool is at capacity; a batch-tier request was rejected outright.

    Mapped to HTTP 429 with a Retry-After header by api/errors.
    """

    def __init__(self, message: str, retry_after_s: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of a successful admission check."""

    effective_max_tokens: int
    reserved_tokens: int


def _estimate_prompt_tokens(engine: Any, request: ChatCompletionRequest) -> int:
    """Prompt token count for admission math.

    Prefers the engine's own tokenizer (VLLMEngine.count_prompt_tokens) via
    getattr — mirrors the existing getattr(engine, "kv_capacity_tokens", None)
    pattern in api/routes/metrics.py, since BaseEngine doesn't declare either
    as part of its abstract contract. Falls back to a chars/4 heuristic for
    engines that don't expose a tokenizer (e.g. test stubs).
    """
    counter = getattr(engine, "count_prompt_tokens", None)
    if callable(counter):
        try:
            return int(counter(request))
        except Exception as exc:
            logger.debug("count_prompt_tokens failed, using chars/4 fallback: %s", exc)
    total_chars = sum(len(m.content) for m in request.messages)
    return max(1, total_chars // _CHARS_PER_TOKEN_FALLBACK)


class _KVReservationTracker:
    """Thread-safe per-model counter of tokens reserved by in-flight requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reserved: dict[str, int] = {}

    def current(self, model: str) -> int:
        with self._lock:
            return self._reserved.get(model, 0)

    def reserve(self, model: str, tokens: int) -> None:
        with self._lock:
            self._reserved[model] = self._reserved.get(model, 0) + tokens

    def release(self, model: str, tokens: int) -> None:
        with self._lock:
            self._reserved[model] = max(0, self._reserved.get(model, 0) - tokens)


class _InFlightSeqTracker:
    """Thread-safe per-model counter of in-flight admitted requests (sequence slots).

    Unlike _KVReservationTracker (token counts), this counts requests, since
    vLLM's max_num_seqs is a hard cap on concurrent sequences regardless of how
    few tokens each one uses.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def current(self, model: str) -> int:
        with self._lock:
            return self._counts.get(model, 0)

    def increment(self, model: str) -> None:
        with self._lock:
            self._counts[model] = self._counts.get(model, 0) + 1

    def decrement(self, model: str) -> None:
        with self._lock:
            self._counts[model] = max(0, self._counts.get(model, 0) - 1)


class AdmissionController:
    """Enforces context-length and KV-budget limits before an engine is invoked."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        tier: Any | None = None,
        kv_safety_margin: float = _DEFAULT_KV_SAFETY_MARGIN,
    ) -> None:
        self._registry = registry
        self._tier = tier
        self._kv_safety_margin = kv_safety_margin
        self._tracker = _KVReservationTracker()
        self._seq_tracker = _InFlightSeqTracker()

    def _context_ceiling(self, routed_model: str) -> int:
        """Highest token count (prompt + output) this model's context window allows."""
        ceiling = self._tier.max_model_len_cap if self._tier is not None else _DEFAULT_CONTEXT_CAP
        if routed_model in self._registry:
            entry_cap = self._registry.get(routed_model).max_model_len
            if entry_cap is not None:
                ceiling = min(ceiling, entry_cap)
        return ceiling

    def _effective_max_num_seqs(self, routed_model: str) -> int | None:
        """Resolved sequence-concurrency ceiling for *routed_model*, or None if no
        tier is resolved (gate is skipped entirely — fail open, same posture as
        the context/KV gates when their underlying numbers are unavailable)."""
        if self._tier is None:
            return None
        ceiling = self._tier.max_num_seqs
        if routed_model in self._registry:
            entry_cap = self._registry.get(routed_model).max_num_seqs
            if entry_cap is not None:
                ceiling = min(ceiling, entry_cap)
        return ceiling

    def admit(
        self,
        routed_model: str,
        request: ChatCompletionRequest,
        engine: Any,
    ) -> AdmissionResult:
        """Validate *request* against context and KV limits for *routed_model*.

        Returns the admitted (possibly clamped) max_tokens and the token count
        reserved against the KV tracker — the caller MUST call release() with
        that same reserved_tokens value once the request completes (success or
        failure), typically from a try/finally around engine dispatch.

        Raises:
            ContextTooLongError: prompt alone exceeds the context ceiling, or
                requested_output can't be clamped to a usable size (batch tier,
                or interactive with essentially no room left).
            EngineSaturatedError: KV pool is saturated and the request is
                batch-tier (429; caller should retry later), or the model's
                sequence-concurrency ceiling is already full (429 for either
                priority — there is no clamp path for a sequence slot).
        """
        effective_max_num_seqs = self._effective_max_num_seqs(routed_model)
        if effective_max_num_seqs is not None:
            in_flight = self._seq_tracker.current(routed_model)
            if in_flight >= effective_max_num_seqs:
                raise EngineSaturatedError(
                    f"Model '{routed_model}' is at its concurrent-sequence limit "
                    f"({in_flight}/{effective_max_num_seqs} in flight); retry shortly.",
                    retry_after_s=1.0,
                )

        requested_output = request.max_output_tokens or request.max_tokens or 512
        context_ceiling = self._context_ceiling(routed_model)
        if request.max_context_tokens is not None:
            context_ceiling = min(context_ceiling, request.max_context_tokens)

        prompt_tokens = _estimate_prompt_tokens(engine, request)
        if prompt_tokens > context_ceiling:
            raise ContextTooLongError(
                f"Prompt is {prompt_tokens} tokens, which exceeds the "
                f"{context_ceiling}-token context limit for model '{routed_model}'."
            )

        effective_output = requested_output
        if prompt_tokens + effective_output > context_ceiling:
            room = context_ceiling - prompt_tokens
            if request.priority == "batch" or room < _MIN_CLAMPED_OUTPUT_TOKENS:
                raise ContextTooLongError(
                    f"Prompt ({prompt_tokens} tokens) + requested output "
                    f"({effective_output} tokens) exceeds the {context_ceiling}-token "
                    f"context limit for model '{routed_model}'."
                )
            effective_output = room

        capacity = getattr(engine, "kv_capacity_tokens", None)
        if capacity is not None:
            budget = capacity * self._kv_safety_margin
            reserved_now = self._tracker.current(routed_model)
            available = budget - reserved_now - prompt_tokens
            if available < effective_output:
                if request.priority == "batch" or available < _MIN_CLAMPED_OUTPUT_TOKENS:
                    raise EngineSaturatedError(
                        f"Model '{routed_model}' KV pool is at capacity "
                        f"({reserved_now}/{int(budget)} tokens reserved); retry shortly.",
                        retry_after_s=1.0,
                    )
                effective_output = max(_MIN_CLAMPED_OUTPUT_TOKENS, int(available))

        reserved_tokens = prompt_tokens + effective_output
        self._tracker.reserve(routed_model, reserved_tokens)
        self._seq_tracker.increment(routed_model)
        return AdmissionResult(effective_max_tokens=effective_output, reserved_tokens=reserved_tokens)

    def release(self, routed_model: str, reserved_tokens: int) -> None:
        """Release a reservation made by admit() once the request has completed."""
        self._tracker.release(routed_model, reserved_tokens)
        self._seq_tracker.decrement(routed_model)
