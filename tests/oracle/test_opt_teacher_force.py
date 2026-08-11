"""Gated real-GPU oracle suite (Phase C — C4a).

Skipped unless ``INFERENCE_X_RUN_GPU_TESTS=1`` and CUDA is available. Not part of
the GPU-free ``checks`` job (which runs ``tests/unit`` only).
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("INFERENCE_X_RUN_GPU_TESTS") != "1",
        reason="set INFERENCE_X_RUN_GPU_TESTS=1 to run gated GPU oracle tests",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA GPU required for oracle suite",
    ),
]


ORACLE_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Inference correctness is measured token by token."
)
MODEL_NAME = "opt-125m"
MODEL_PATH = "facebook/opt-125m"
PROMPT_LOGPROBS_TOP = 5


def _top2_from_logits(logits_row) -> tuple[int, float, int, float]:
    """Return (top1_id, top1_logit, top2_id, top2_logit) from a 1-D logit vector."""
    values, indices = torch.topk(logits_row, k=min(2, logits_row.numel()))
    top1_id = int(indices[0].item())
    top1_logit = float(values[0].item())
    if values.numel() < 2:
        return top1_id, top1_logit, top1_id, top1_logit
    return top1_id, top1_logit, int(indices[1].item()), float(values[1].item())


def _transformers_scores(token_ids: list[int]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inference_x.utils.oracle_compare import PositionScore

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Keep the HF tokenizer aligned with the ids we already scored on the engine
    # side; re-check round-trip for sanity.
    roundtrip = tokenizer.convert_ids_to_tokens(token_ids)
    assert roundtrip, "empty tokenisation"

    input_ids = torch.tensor([token_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq, vocab]

    scores: list[PositionScore] = []
    # Position i (0-based over predicted tokens) uses logits[i] to predict token i+1.
    for i in range(len(token_ids) - 1):
        top1_id, top1_logit, top2_id, top2_logit = _top2_from_logits(logits[i])
        scores.append(
            PositionScore(
                top1_id=top1_id,
                top1_logit=top1_logit,
                top2_id=top2_id,
                top2_logit=top2_logit,
            )
        )
    return scores


async def _engine_scores(token_ids: list[int], prompt_text: str):
    import uuid

    from inference_x.utils.cuda_env import ensure_vllm_runtime_env

    ensure_vllm_runtime_env()

    from vllm import SamplingParams

    from inference_x.engines.vllm_engine import VLLMEngine
    from inference_x.utils.oracle_compare import PositionScore

    engine = VLLMEngine(
        {
            "name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "max_model_len": 256,
            "max_completion_tokens": 1,
            "instruction_tuned": False,
            "gpu_memory_utilization": 0.4,
        }
    )
    try:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=PROMPT_LOGPROBS_TOP,
        )
        request_id = f"oracle-{uuid.uuid4().hex}"
        final = None
        async for output in engine._llm.generate(prompt_text, sampling, request_id):
            final = output
            if output.finished:
                break
        assert final is not None
        assert final.prompt_logprobs is not None

        scores: list[PositionScore] = []
        # prompt_logprobs[0] is typically None (no prediction for the first token).
        # prompt_logprobs[i] scores token_ids[i] given the prefix; aligns with
        # transformers logits[i-1] predicting token_ids[i].
        for i in range(1, len(token_ids)):
            entry = final.prompt_logprobs[i]
            assert entry is not None, f"missing prompt_logprobs at {i}"
            # Prefer the logprob dict's ranked entries; fall back to sorting.
            ranked = sorted(
                (
                    (int(tid), float(getattr(lp, "logprob", lp)))
                    for tid, lp in entry.items()
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )
            assert ranked, f"empty prompt_logprobs at {i}"
            top1_id, top1_lp = ranked[0]
            top2_id, top2_lp = ranked[1] if len(ranked) > 1 else (top1_id, top1_lp)
            scores.append(
                PositionScore(
                    top1_id=top1_id,
                    top1_logit=top1_lp,
                    top2_id=top2_id,
                    top2_logit=top2_lp,
                )
            )
        return scores
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


@pytest.mark.asyncio
async def test_opt125m_teacher_forced_top1_vs_transformers(capsys):
    from transformers import AutoTokenizer

    from inference_x.utils.oracle_compare import classify_positions, format_report

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    token_ids = tokenizer.encode(ORACLE_PROMPT, add_special_tokens=True)
    assert len(token_ids) >= 8, "prompt too short for a meaningful oracle"

    engine_scores = await _engine_scores(token_ids, ORACLE_PROMPT)
    reference_scores = _transformers_scores(token_ids)
    assert len(engine_scores) == len(reference_scores) == len(token_ids) - 1

    report = classify_positions(engine_scores, reference_scores)
    print(format_report(report))
    captured = capsys.readouterr()
    assert "oracle:" in captured.out
    if report.near_ties:
        assert "near_tie@" in captured.out

    assert report.passed, (
        "hard mismatches: "
        + ", ".join(
            f"@{v.index} engine={v.engine.top1_id} ref={v.reference.top1_id}"
            for v in report.mismatches
        )
    )
