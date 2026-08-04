"""OS-3 seed identity / concurrency tests (DEC-051).

Identity asserts identical content under sequential execution with the same
prompt, parameters, and seed — a validation of forwarding, not an end-to-end
runtime guarantee (G5 / N1). Concurrency remains xfail citing Phase C3.
"""

from __future__ import annotations

import pytest

from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatStreamChunk,
)


def _seeded_content(request: ChatCompletionRequest) -> str:
    """Deterministic stand-in for a backend that honours seed (forwarding check)."""
    prompt = "|".join(f"{m.role}:{m.content}" for m in request.messages)
    return (
        f"seed={request.seed}|temp={request.temperature}|top_p={request.top_p}"
        f"|prompt={prompt}"
    )


class _SeedHonouringEngine(BaseEngine):
    """Stub backend that honours request.seed in produced content."""

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        text = _seeded_content(request)
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        )

    async def generate_stream(self, request: ChatCompletionRequest):
        yield ChatStreamChunk(content=_seeded_content(request))
        yield ChatStreamChunk(content="", finish_reason="stop")

    def is_healthy(self) -> bool:
        return True


def _request(*, seed: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="ping")],
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        seed=seed,
    )


@pytest.mark.asyncio
async def test_sequential_same_seed_identical_content():
    """Same prompt, params, seed, sequential → identical content (forwarding check)."""
    engine = _SeedHonouringEngine()
    req = _request(seed=42)

    first = await engine.generate(req)
    second = await engine.generate(req)

    assert first.choices[0].message.content == second.choices[0].message.content
    assert first.choices[0].message.content.startswith("seed=42|")


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Phase C3: batch-composition nondeterminism under concurrency; "
    "OS-3 does not guarantee identity for multi-in-flight seeded requests (N1).",
    strict=True,
)
async def test_concurrent_same_seed_not_guaranteed():
    """Placeholder until Phase C3; must not be treated as an OS-3 guarantee."""
    pytest.fail(
        "Phase C3: concurrent seeded identity is intentionally non-guaranteed (N1)"
    )
