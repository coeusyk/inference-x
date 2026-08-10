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
   engine-construction time). Enforced by a per-model asyncio.BoundedSemaphore:
   a request waits up to admission_wait_s for a slot to free rather than being
   rejected instantly, since vLLM's own scheduler queues (rather than rejects)
   past this ceiling too (see openspec/changes/rescope-admission-control).
   Unlike the two gates below, there is no clamp path: a request either gets a
   sequence slot within the wait or it doesn't, so both 'interactive' and
   'batch' priority get EngineSaturatedError when the wait elapses.

2. Context length: prompt_tokens + requested_output_tokens must fit the
   model's context window (min of ModelEntry.max_model_len, the resolved VRAM
   tier's max_model_len_cap, and the request's own max_context_tokens, if
   set). Prompt tokens come from BaseEngine.count_prompt_tokens(), the declared
   capability (DEC-047 §3); when it returns None a chars/4 heuristic is used.

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

Since OS-4 that degradation is typed rather than silent: every skipped gate and
every estimated input produces a ResponseWarning of type "degraded" alongside a
structured log record, and every clamp produces one of type "substituted". The
warnings ride out on the response so a client can see what the server did
(PHASE-A-EXECUTION-PLAN §9 C.6/C.8). Making it observable is the whole change —
what admission *decides* is unchanged, and no degraded condition ever rejects,
because that would turn fail-open into fail-closed (DEC-047 §4).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any

from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import ChatCompletionRequest, ResponseWarning
from inference_x.services.model_service import ModelRegistry

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_CAP = 4096  # matches the old ChatService._GLOBAL_MAX_TOKENS default
_DEFAULT_KV_SAFETY_MARGIN = 0.9  # leave headroom below the real KV pool size
_MIN_CLAMPED_OUTPUT_TOKENS = 16  # below this a "clamp" is really a rejection in disguise
_CHARS_PER_TOKEN_FALLBACK = 4  # used only when the engine has no tokenizer to ask
_DEFAULT_ADMISSION_WAIT_S = 5.0  # fallback only; production value comes from settings


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


class StrictModeViolationError(ValueError):
    """The request asked for `strict` and the server would have substituted.

    Subclasses ValueError so it is handled by the existing sanitized 400 handler
    (api/errors.value_error_handler) without a new registration — the same route
    ContextTooLongError takes.

    Raised only where a "substituted" warning would otherwise be emitted, from
    inside that same branch (DEC-052: one predicate, two outcomes). Never raised
    for a "degraded" condition — see the module docstring.
    """


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of a successful admission check."""

    effective_max_tokens: int
    reserved_tokens: int
    warnings: tuple[ResponseWarning, ...] = ()


def _warn(
    sink: list[ResponseWarning],
    *,
    type: str,
    code: str,
    message: str,
    field_name: str | None = None,
) -> None:
    """Append a warning and log the same condition — one mechanism, two sinks.

    Building the structured log and the response warning as two independent
    mechanisms guarantees the second one drifts from the first, so they are
    emitted here together or not at all (PHASE-A-EXECUTION-PLAN §3.4c).
    """
    sink.append(
        ResponseWarning(type=type, code=code, message=message, field=field_name)  # type: ignore[arg-type]
    )
    logger.info("admission %s: code=%s field=%s %s", type, code, field_name, message)


def _prompt_tokens(
    engine: BaseEngine,
    request: ChatCompletionRequest,
    warnings: list[ResponseWarning],
) -> int:
    """Prompt token count for admission math, via the declared capability.

    Calls BaseEngine.count_prompt_tokens() — declared since OS-4 per DEC-047 §3,
    rather than probed with getattr. `None` means this engine has no tokenizer to
    ask, which is a supported state: the chars/4 heuristic stands in and the
    substitution is reported instead of being silently absorbed.

    This heuristic is an admission *gate input*, never a reported figure — the
    one authorized use of character-based estimation left in the codebase
    (PHASE-A-EXECUTION-PLAN §9 A.3).
    """
    counted = engine.count_prompt_tokens(request)
    if counted is not None:
        return int(counted)
    total_chars = sum(len(m.content) for m in request.messages)
    estimate = max(1, total_chars // _CHARS_PER_TOKEN_FALLBACK)
    _warn(
        warnings,
        type="degraded",
        code="prompt_tokens_estimated",
        field_name="messages",
        message=(
            f"Engine reported no prompt-token count; admission used a "
            f"chars/{_CHARS_PER_TOKEN_FALLBACK} estimate of {estimate} tokens."
        ),
    )
    return estimate


class _PerModelCounter:
    """Thread-safe per-model counter, shared shape for both KV-token and
    in-flight-sequence tracking below (only the unit each counts differs:
    reserved tokens vs. admitted requests)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def current(self, model: str) -> int:
        with self._lock:
            return self._counts.get(model, 0)

    def add(self, model: str, amount: int) -> None:
        with self._lock:
            self._counts[model] = max(0, self._counts.get(model, 0) + amount)


class AdmissionController:
    """Enforces context-length and KV-budget limits before an engine is invoked."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        tier: Any | None = None,
        kv_safety_margin: float = _DEFAULT_KV_SAFETY_MARGIN,
        admission_wait_s: float = _DEFAULT_ADMISSION_WAIT_S,
    ) -> None:
        self._registry = registry
        self._tier = tier
        self._kv_safety_margin = kv_safety_margin
        self._admission_wait_s = admission_wait_s
        self._tracker = _PerModelCounter()
        self._semaphores: dict[str, asyncio.BoundedSemaphore] = {}

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

    def _get_semaphore(self, routed_model: str, max_num_seqs: int) -> asyncio.BoundedSemaphore:
        """Lazily create *routed_model*'s sequence-concurrency semaphore.

        Sized once, at first use, from the tier-resolved max_num_seqs — safe
        because that ceiling is a per-instance constant (no dynamic model
        load/unload exists today, see design.md "Semaphore ownership and
        keying"). No `await` happens between the lookup and the insert, so
        this is race-free under the single-process, no-`--workers` deployment
        this codebase runs (see design.md "Concurrency / race analysis").
        """
        sem = self._semaphores.get(routed_model)
        if sem is None:
            sem = asyncio.BoundedSemaphore(max_num_seqs)
            self._semaphores[routed_model] = sem
        return sem

    async def admit(
        self,
        routed_model: str,
        request: ChatCompletionRequest,
        engine: BaseEngine,
    ) -> AdmissionResult:
        """Validate *request* against context and KV limits for *routed_model*.

        Returns the admitted (possibly clamped) max_tokens, the token count
        reserved against the KV tracker, and every warning describing what this
        method substituted or could not check. The caller MUST call release()
        with that same reserved_tokens value once the request completes (success
        or failure), typically from a try/finally around engine dispatch.

        This method is a coroutine because the sequence-concurrency gate may
        wait (bounded by admission_wait_s) for a slot to free rather than
        rejecting instantly — vLLM's own scheduler queues past this ceiling
        too, so an instant reject was a false 429 in cases where a slot was
        about to free. Gates 2/3 remain fully synchronous; the only
        suspension point in this method is the sequence-concurrency wait.

        Raises:
            ContextTooLongError: prompt alone exceeds the context ceiling, or
                requested_output can't be clamped to a usable size (batch tier,
                or interactive with essentially no room left).
            EngineSaturatedError: KV pool is saturated and the request is
                batch-tier (429; caller should retry later), or the model's
                sequence-concurrency ceiling stayed full for longer than
                admission_wait_s (429 for either priority — there is no clamp
                path for a sequence slot).
            StrictModeViolationError: request.strict is set and this method would
                otherwise have clamped a parameter (400).
        """
        warnings: list[ResponseWarning] = []

        effective_max_num_seqs = self._effective_max_num_seqs(routed_model)
        sem: asyncio.BoundedSemaphore | None = None
        if effective_max_num_seqs is None:
            _warn(
                warnings,
                type="degraded",
                code="sequence_gate_skipped",
                message=(
                    "No VRAM tier resolved; the sequence-concurrency gate did not "
                    "run for this request."
                ),
            )
        else:
            sem = self._get_semaphore(routed_model, effective_max_num_seqs)
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self._admission_wait_s)
            except TimeoutError:
                raise EngineSaturatedError(
                    f"Model '{routed_model}' is at its concurrent-sequence limit; "
                    f"no slot freed within {self._admission_wait_s}s; retry shortly.",
                    retry_after_s=1.0,
                )

        # Gates 2/3 + final commit are wrapped so a rejection here releases the
        # sequence-concurrency permit acquired above (lifecycle path 7 — see
        # design.md "Semaphore lifecycle analysis"). Safe as a blanket release
        # only because Gates 2/3 contain no `await` below — if that ever
        # changes, this must become an explicit `acquired` flag guard instead.
        try:
            requested_output = request.max_output_tokens or request.max_tokens or 512
            context_ceiling = self._context_ceiling(routed_model)
            if request.max_context_tokens is not None:
                context_ceiling = min(context_ceiling, request.max_context_tokens)

            prompt_tokens = _prompt_tokens(engine, request, warnings)
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
                # DEC-052: one predicate, two outcomes. The condition that raises under
                # strict is textually the condition that warns by default — there is no
                # second `if` for the two to drift apart on.
                if request.strict:
                    raise StrictModeViolationError(
                        f"strict: requested output ({effective_output} tokens) does not fit "
                        f"the {context_ceiling}-token context limit for model "
                        f"'{routed_model}' with a {prompt_tokens}-token prompt; "
                        f"the server would have clamped it to {room}."
                    )
                _warn(
                    warnings,
                    type="substituted",
                    code="max_tokens_clamped_to_context",
                    field_name="max_tokens",
                    message=(
                        f"Requested {effective_output} output tokens; clamped to {room} to "
                        f"fit the {context_ceiling}-token context limit."
                    ),
                )
                effective_output = room

            capacity = getattr(engine, "kv_capacity_tokens", None)
            if capacity is None:
                _warn(
                    warnings,
                    type="degraded",
                    code="kv_gate_skipped",
                    message=(
                        "Engine reported no KV capacity; the KV-pressure gate did not "
                        "run for this request."
                    ),
                )
            else:
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
                    clamped = max(_MIN_CLAMPED_OUTPUT_TOKENS, int(available))
                    if request.strict:
                        raise StrictModeViolationError(
                            f"strict: model '{routed_model}' has {int(available)} tokens of "
                            f"KV budget available for {effective_output} requested output "
                            f"tokens; the server would have clamped it to {clamped}."
                        )
                    _warn(
                        warnings,
                        type="substituted",
                        code="max_tokens_clamped_to_kv_budget",
                        field_name="max_tokens",
                        message=(
                            f"Requested {effective_output} output tokens; clamped to "
                            f"{clamped} to fit the available KV budget."
                        ),
                    )
                    effective_output = clamped

            reserved_tokens = prompt_tokens + effective_output
            self._tracker.add(routed_model, reserved_tokens)
            return AdmissionResult(
                effective_max_tokens=effective_output,
                reserved_tokens=reserved_tokens,
                warnings=tuple(warnings),
            )
        except BaseException:
            if sem is not None:
                sem.release()
            raise

    def release(self, routed_model: str, reserved_tokens: int) -> None:
        """Release a reservation made by admit() once the request has completed.

        Releases the sequence-concurrency permit only if one was acquired for
        this model (i.e. a semaphore exists for it — absent only when the
        sequence gate is skipped entirely because no VRAM tier is resolved,
        see _effective_max_num_seqs).
        """
        self._tracker.add(routed_model, -reserved_tokens)
        sem = self._semaphores.get(routed_model)
        if sem is not None:
            sem.release()
