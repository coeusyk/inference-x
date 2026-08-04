from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=32_000)


class StreamOptions(BaseModel):
    """OpenAI-compatible `stream_options` object (DEC-049).

    Only `include_usage` is supported. An absent `stream_options` is equivalent
    to `include_usage=False`.
    """

    include_usage: bool = Field(
        default=False,
        description="When true, the stream emits a final chunk with an empty "
        "choices array carrying engine-accounted usage, before data: [DONE].",
    )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    max_context_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Prompt-token ceiling enforced pre-dispatch by AdmissionController; "
        "None means the model's configured max_model_len is the only ceiling.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Preferred alias for max_tokens; takes precedence over max_tokens "
        "when both are set. Kept max_tokens for OpenAI-client compatibility.",
    )
    priority: Literal["interactive", "batch"] = Field(
        default="interactive",
        description="'interactive' requests get output clamped to fit available context/"
        "KV budget when possible; 'batch' requests are rejected (429) instead of clamped "
        "when the engine is saturated. Client-supplied and unauthenticated — do not treat "
        "as a trust boundary.",
    )
    stream_options: Optional[StreamOptions] = Field(
        default=None,
        description="OpenAI-compatible streaming options. Absent means "
        "include_usage=False — no usage chunk is emitted.",
    )


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "error"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatStreamChunk(BaseModel):
    """One event on the engine streaming channel (DEC-049).

    This is the `BaseEngine.generate_stream` element type — the Engine Boundary's
    streaming vocabulary. It deliberately lives in `schemas.chat` beside the wire
    models: DEC-047 forbids a backend-neutral execution package, and a second
    type system for the same information is exactly what that prohibits.

    Content events set `content` and leave `finish_reason` and `usage` None.
    The terminal event sets `finish_reason`, and sets `usage` when the backend
    can account it. Per-request timings are Phase B3 and are not carried here.
    """

    content: str = ""
    finish_reason: Optional[Literal["stop", "length", "error"]] = None
    usage: Optional[ChatCompletionUsage] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
