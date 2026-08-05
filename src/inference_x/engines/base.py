from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from inference_x.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatStreamChunk,
)


class BaseEngine(ABC):
    """The Engine Boundary. All inference implementations must satisfy this interface.

    The rest of the system depends on this interface, not on concrete
    implementations: services, routing, and observability never import a backend.

    This contract is stable, not frozen. An earlier version of this docstring
    claimed that adding a second engine "must not require changes here" — that
    was optimistic, and DEC-047's problem statement said so. DEC-049 widened
    `generate_stream` because a bare `str` could not carry the usage a truthful
    metric requires. What the contract does guarantee is narrower and honest:

    - It speaks `schemas.chat` wire types. There is no parallel execution type
      system and no `inference_x/execution/` package (DEC-047).
    - Widening it requires an accepted ADR that says what changed and why.
    - It declares `count_prompt_tokens`, which DEC-047 §3 names durable. It does
      **not** declare `kv_capacity_tokens`, which DEC-047 §3 marks provisional
      and forbids freezing as a cross-backend contract — that one stays an
      optional attribute its callers discover with `getattr`.

    Backends own inference execution. The runtime owns architectural policy.
    """

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run inference for a chat completion request and return a typed response."""

    @abstractmethod
    def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Run inference and yield chunks as they are generated (DEC-049).

        Content events carry `content` with `finish_reason` and `usage` None.
        The final event carries `finish_reason`, and carries `usage` when the
        backend can account it — implementations that cannot must leave `usage`
        None rather than estimate it.
        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if the engine is loaded and ready to serve requests."""

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int | None:
        """Prompt token count for admission, or None when this engine can't count.

        Declared per DEC-047 §3 (durable capability) so `routing/admission.py`
        calls a contract instead of probing with `getattr`.

        Deliberately **not** abstract, and deliberately returning `int | None`.
        An engine without a tokenizer is a supported state, not an error: callers
        must fall back and continue rather than reject (DEC-047 §4). Making this
        required would remove "unavailable" as a representable state, which is
        the one state the fail-open posture is built on.
        """
        return None
