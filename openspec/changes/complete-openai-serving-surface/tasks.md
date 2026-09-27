# Tasks — complete-openai-serving-surface (V1-0)

## 1. Request surface
- [x] 1.1 `extra="forbid"` on `ChatCompletionRequest` and `ChatMessage` (D1)
- [x] 1.2 Bounds: messages ≤ 2048, content ≤ 1,000,000, `max_tokens`/`max_output_tokens`
      default None, no upper bound (D4, D5)
- [x] 1.3 `stop` field + validation (D3); `include_manifest` field + stream conflict validator (D7)
- [x] 1.4 `ResolvedRequest.stop`; schema derivability test exclusion set gains `include_manifest`

## 2. Errors
- [x] 2.1 `ErrorDetail.param`; `RequestValidationError` → 400 envelope (D2)
- [x] 2.2 `ContextTooLongError` → 400 `context_length_exceeded`, OpenAI-form message;
      `StrictModeViolationError` → 400 `strict_violation` (D6)

## 3. Admission + tiers
- [x] 3.1 `effective_context()` helper in `utils/vram_tiers.py` + unit tests (D8)
- [x] 3.2 `_context_ceiling` uses it; omitted-output → remaining context, no warning (D4)
- [x] 3.3 `/v1/models` and `/v1/plan` report effective context / seqs / composed / tier_limited

## 4. Engine + service
- [x] 4.1 `stop` into `SamplingParams`; `ManifestSampling.stop`
- [x] 4.2 Opt-in manifest in `ChatService.complete`; route omits absent `manifest` key;
      port #38's run_id-recomputation test behind the opt-in
- [x] 4.3 HF preflight fails open offline / on transport errors (D10)
- [x] 4.4 `max_completion_tokens` resolved in admission, removed from `VLLMEngine` (D11);
      regression tests in test_admission, test_vllm_sampling, test_routes

## 5. Config + docs
- [x] 5.1 `qwen2.5-coder-1.5b` entry in `config/models.yaml` (D9)
- [x] 5.2 DEC-063 and DEC-064 in `docs/DECISIONS.md`
- [x] 5.3 CHANGELOG `[Unreleased]`
- [x] 5.4 README Aider quickstart (the exact invocation used by 6.4); `scripts/aider_acceptance.py`;
      `scripts/smoke_test.py --model`

## 6. Validation
- [x] 6.1 Unit tests for every ADDED requirement scenario; full suite (668 passed, 1 xfailed), ruff, mypy green
- [x] 6.2 `openspec validate complete-openai-serving-surface --strict`
- [x] 6.3 Live (RTX 3060 Laptop 6 GiB, vLLM 0.22.1, qwen2.5-0.5b composed 8192×1, offline):
      /v1/plan + /v1/models report composed 8192×1 and tier-limited 2048 for the plain entry;
      measured KV 97,824 tokens; stop, omitted max_tokens (8151 resolved, no warning),
      opt-in manifest (run_id == X-Run-Id == manifest.run_id), context_length_exceeded,
      unknown-field 400, streaming finish_reason + usage — all verified
- [x] 6.4 Live: real Aider session completes a correct edit. Laptop, earlier: protocol verified with
      qwen2.5-0.5b (3 sessions, 0 server errors) but that model could not follow Aider's edit formats.
      Desktop 2026-09-27 (RTX 4060 8 GiB, 6gb tier, vLLM 0.22.1, Aider 0.86.2, qwen2.5-coder-1.5b
      composed 8192x1, measured KV 52,384 tokens, HF_HUB_OFFLINE=1): `scripts/aider_acceptance.py`
      passed 7/7 (4 streaming, 3 `--no-stream`, `whole` format); the buggy test fails before the edit
      and passes after. Logging proxy: Aider sends only model/temperature/stream + role/content
      messages, all 200, no warnings, non-stream `resolved.max_tokens` 7394 (remaining context),
      streams end with finish_reason stop + [DONE]; smoke_test passes
- [x] 6.5 Live: `include_manifest` round-trip; `X-Run-Id` present by default
- [x] 6.6 Live (opt-125m, cap 256): `max_tokens` 512 resolves to 256 with
      `max_tokens_clamped_to_model_cap`; `max_tokens` 8 resolves to 8 and runs exactly 8 tokens
      (finish_reason length); omitted resolves to 256 with no warning; strict over-cap is 400
      `strict_violation`; manifest `resolved_max_tokens` matches `resolved` (D11)
