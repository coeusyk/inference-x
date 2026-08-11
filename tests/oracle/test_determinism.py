"""Gated determinism conformance (Phase C — C4b).

Pass criterion: N seeded trials under ``deterministic: true`` produce exactly
one unique sample. Skipped unless ``INFERENCE_X_RUN_GPU_TESTS=1`` and CUDA.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("INFERENCE_X_RUN_GPU_TESTS") != "1",
        reason="set INFERENCE_X_RUN_GPU_TESTS=1 to run gated GPU determinism tests",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA GPU required for determinism suite",
    ),
]

N_TRIALS = 5
PROMPT = "Say only the word ping."
MODEL_NAME = "opt-125m"
MODEL_PATH = "facebook/opt-125m"


@pytest.mark.asyncio
async def test_deterministic_trials_produce_one_unique_sample():
    from inference_x.engines.vllm_engine import VLLMEngine
    from inference_x.schemas.chat import ChatCompletionRequest, ChatMessage
    from inference_x.utils.cuda_env import ensure_vllm_runtime_env
    from inference_x.utils.determinism import (
        ensure_deterministic_mode,
        warm_deterministic_engine,
    )

    ensure_vllm_runtime_env()
    ensure_deterministic_mode(require=True)

    engine = VLLMEngine(
        {
            "name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "max_model_len": 256,
            "max_completion_tokens": 16,
            "instruction_tuned": False,
            "gpu_memory_utilization": 0.4,
        }
    )
    try:
        await warm_deterministic_engine(engine, iterations=3)
        request = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[ChatMessage(role="user", content=PROMPT)],
            max_tokens=8,
            temperature=0.0,
            seed=42,
            deterministic=True,
        )
        samples: list[str] = []
        for _ in range(N_TRIALS):
            response = await engine.generate(request)
            samples.append(response.choices[0].message.content)
        unique = set(samples)
        assert len(unique) == 1, (
            f"expected 1 unique sample over {N_TRIALS} trials, got "
            f"{len(unique)}: {unique!r}"
        )
    finally:
        engine.shutdown()
