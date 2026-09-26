# Tasks — complete-openai-serving-surface (V1-0)

## 1. Request surface
- [ ] 1.1 `extra="forbid"` on `ChatCompletionRequest` and `ChatMessage` (D1)
- [ ] 1.2 Bounds: messages ≤ 2048, content ≤ 1,000,000, `max_tokens`/`max_output_tokens`
      default None, no upper bound (D4, D5)
- [ ] 1.3 `stop` field + validation (D3); `include_manifest` field + stream conflict validator (D7)
- [ ] 1.4 `ResolvedRequest.stop`; schema derivability test exclusion set gains `include_manifest`

## 2. Errors
- [ ] 2.1 `ErrorDetail.param`; `RequestValidationError` → 400 envelope (D2)
- [ ] 2.2 `ContextTooLongError` → 400 `context_length_exceeded`, OpenAI-form message;
      `StrictModeViolationError` → 400 `strict_violation` (D6)

## 3. Admission + tiers
- [ ] 3.1 `effective_context()` helper in `utils/vram_tiers.py` + unit tests (D8)
- [ ] 3.2 `_context_ceiling` uses it; omitted-output → remaining context, no warning (D4)
- [ ] 3.3 `/v1/models` and `/v1/plan` report effective context / seqs / composed / tier_limited

## 4. Engine + service
- [ ] 4.1 `stop` into `SamplingParams`; `ManifestSampling.stop`
- [ ] 4.2 Opt-in manifest in `ChatService.complete`; route omits absent `manifest` key;
      port #38's run_id-recomputation test behind the opt-in

## 5. Config + docs
- [ ] 5.1 `qwen2.5-coder-1.5b` entry in `config/models.yaml` (D9)
- [ ] 5.2 DEC-063 (API surface: forbid/400/bounds/max_tokens/manifest opt-in; supersedes
      DEC-025 caps) and DEC-064 (tier KV-budget composition) in `docs/DECISIONS.md`
- [ ] 5.3 CHANGELOG `[Unreleased]`; README Aider quickstart

## 6. Validation
- [ ] 6.1 Unit tests for every ADDED requirement scenario; full suite, ruff, mypy green
- [ ] 6.2 `openspec validate complete-openai-serving-surface --strict`
- [ ] 6.3 Live: boot `qwen2.5-coder-1.5b` on the 6 GiB laptop; record plan output, KV capacity
- [ ] 6.4 Live: real Aider session (streaming, whole + diff formats) completes an edit;
      record incompatibilities found, if any
- [ ] 6.5 Live: `include_manifest` round-trip; `X-Run-Id` present by default
