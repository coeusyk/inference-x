# Design: complete-openai-serving-surface (V1-0)

## Context

Evidence comes from Aider 0.86.2 traffic captured against a logging stub
(2026-09-26), plus the audit recorded in `docs/INFERENCEX-EVOLUTION.md` §6.
Invariants preserved:

- DEC-047 Engine Boundary
- B6 one model per process
- `AsyncLLM`
- `EngineTiming`
- `run_id` / `X-Run-Id`
- `strict` / `warnings` / `resolved` semantics
- never fabricate

## D1 — Reject unknown fields (`extra="forbid"`)

`ChatCompletionRequest` and `ChatMessage` use `model_config = ConfigDict(extra="forbid")`.
The alternatives were considered and rejected:

- **An allowlist of "harmless" fields that are silently ignored.** Rejected: whether a
  field is harmless is a claim about client intent that the server cannot verify.
- **Accepting default-valued no-ops such as `n: 1`.** Rejected: no acceptance client
  has shown it sends them (YAGNI).

Continue validation is the evidence gate for adding any field. Varex sends only
`model`/`messages`/`max_tokens` and is unaffected.

## D2 — OpenAI error envelope for validation

A `RequestValidationError` handler maps FastAPI's 422 to a 400 with the existing
`ErrorResponse` shape. `ErrorDetail` gains an optional `param`, which is additive.
`param` is the first error's `loc` with the leading `body` segment dropped, joined
with dots. The message is pydantic's own `msg` prefixed by the location. It exposes
no internals, so unlike the generic `ValueError` handler it is not sanitized. This
changes the status code for every validation error (422 → 400); recorded in DEC-063.

## D3 — `stop`

Type: `Optional[str | list[str]]`. A list holds 1–4 non-empty strings; a single string
must be non-empty. The value is normalized to a list only at the `SamplingParams`
boundary (`stop=[...]`). vLLM's default `include_stop_str_in_output=False` gives the
OpenAI content semantics. `derive_terminal_metadata` already maps vLLM's `"stop"`
finish to `"stop"`.

The field appears in:

- `ResolvedRequest`, as required by the derivability rule;
- `ManifestSampling.stop`, because it changes output and therefore identity.

Engine-boundary check: `stop` is a genuine public OpenAI capability, and llama.cpp's
server supports it too. It is not a vLLM-ism.

## D4 — Omitted `max_tokens` resolves to the remaining context

The schema defaults for `max_tokens` and `max_output_tokens` become `None`; today
`max_tokens` defaults to 512. In `AdmissionController.admit`, the logic becomes:

```text
requested = request.max_output_tokens or request.max_tokens
if requested is None:
    effective_output = context_ceiling - prompt_tokens   # no warning: nothing substituted
    (prompt alone > ceiling or room < _MIN_CLAMPED_OUTPUT_TOKENS → ContextTooLongError)
else:
    existing clamp-with-warning path, unchanged
```

The KV gate then runs unchanged on `effective_output`.

- It can clamp with `max_tokens_clamped_to_kv_budget`. That is truthful: output was
  limited by the KV budget.
- Under `strict`, it rejects exactly as before.

Consequence, accepted: an omitted-`max_tokens` request reserves prompt tokens plus the
remaining context in the KV tracker. That is the honest worst case. On a single-user
coding setup (1 sequence) this has no concurrency cost.

`VLLMEngine._resolve_max_tokens`'s 512 fallback stays, but it is reached only by
callers that bypass admission. `ChatService` always sets the admitted value.

## D5 — Bounds

The limits become:

- `messages`: ≤ 2048 (was 50);
- message `content`: ≤ 1,000,000 chars (was 32,000);
- `max_tokens` and `max_output_tokens`: `ge=1`, with no upper bound (was 4096).

The binding limit is the context gate, which is token-accurate. The remaining caps
only stop pathological bodies from reaching the tokenizer. The server binds loopback
(DEC-DEFER-01). This supersedes DEC-025's caps (DEC-063).

## D6 — Actionable context and strict errors

`ContextTooLongError` messages are rewritten in OpenAI's form, which litellm matches
case-insensitively on the substring "maximum context length is". This lets Aider show
its own context-window guidance:

```text
This model's maximum context length is {ceiling} tokens. However, your messages
resulted in {prompt} tokens[ and you requested {out} output tokens]. ...
```

New dedicated handlers are registered before the `ValueError` handler.

| Error | Status | `error.type` | `error.code` | Message |
|---|---|---|---|---|
| `ContextTooLongError` | 400 | `invalid_request_error` | `context_length_exceeded` | `str(exc)` |
| `StrictModeViolationError` | 400 | `invalid_request_error` | `strict_violation` | `str(exc)` |

Both messages name only model names and token counts. Starlette matches a handler by
the most specific class in the exception's MRO, so both win over `ValueError`.

## D7 — Opt-in manifest

- Request field `include_manifest: bool = False`. It is a transport control, so it is
  excluded from `ResolvedRequest` (the schema test's exclusion set gains it).
- `ChatCompletionResponse.manifest: Optional[RunManifest] = None` (from #38).
- `ChatService.complete` sets `manifest` only when it was requested.
- The route serializes with `model_dump(exclude={"manifest"})` when `manifest` is
  `None`, so the default body has no key. Other `None`-valued fields keep their
  existing `null` serialization.
- A model validator rejects `include_manifest` together with `stream` (400, `param`
  `include_manifest`). The streaming path has no manifest surface (C1 D3), and
  silently ignoring the flag would be a lie.

The opt-in was chosen as a request field rather than a header because it matches the
existing extension controls (`strict`, `deterministic`) and is visible in request
logs.

## D8 — KV-budget tier composition (owner decision 2026-09-26)

A single helper in `utils/vram_tiers.py` serves as the source of truth:

```python
def effective_context(tier, entry_max_model_len, entry_max_num_seqs) -> ContextEnvelope
    # seqs    = min(entry_max_num_seqs or tier.max_num_seqs, tier.max_num_seqs)
    # wanted  = entry_max_model_len or tier.max_model_len_cap
    # if wanted <= cap:                         ctx = wanted, composed = False, tier_limited = False
    # elif wanted * seqs <= cap * tier.seqs:   ctx = wanted, composed = True,  tier_limited = False
    # else:                                     ctx = cap,    composed = False, tier_limited = True
```

`AdmissionController._context_ceiling`, `/v1/models`, and `/v1/plan` all call it.
With `tier=None` the behavior is unchanged: admission uses its default cap, and plan
and models report the entry values.

Arithmetic, checked against the actual vLLM path:

- **The tier cap never reaches vLLM.** `VLLMEngine` passes the entry's own
  `max_model_len` (8192 for `qwen2.5-1.5b`, not 2048) and the tier-composed
  `max_num_seqs` (`apply_tier_knobs`).
- **vLLM sizes its KV pool from `gpu_memory_utilization`**, after weights and
  activation profiling, and reports it as `kv_capacity_tokens`. It is not
  `max_model_len × max_num_seqs`.
- **vLLM refuses to start** if a single sequence of `max_model_len` exceeds that pool.
- **Admission's KV gate reserves per request** against the measured
  `kv_capacity_tokens × kv_safety_margin`.

So `cap × seqs` is a policy envelope, not a VRAM formula. The physical guarantees are
the two measured checks above, and they are unchanged.

`8192 × 1 = 2048 × 4` keeps the tier's worst-case concurrent-token envelope identical.

## D9 — Acceptance model

The new entry is `qwen2.5-coder-1.5b`: `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
`max_model_len: 8192`, `max_num_seqs: 1`, `gpu_memory_utilization: auto`.

- A coder-tuned model is the realistic small-VRAM Aider target.
- About 3.1 GB bf16 weights, which fits the 6 GiB laptop with KV headroom.
- It is a live-validation target only. Unit tests use stubs.

## Risks

- **Clients that relied on 422 or on silent dropping** now get 400. Mitigation: the
  DEC and the CHANGELOG record it.
- **Aider's repo map may exceed 8k tokens on large repos.** The failure is now
  actionable, and Aider's `--map-tokens` bounds it. It is a known consumer-hardware
  limit, not a correctness bug.
- **KV reservation of the full remaining context** (D4) makes concurrent
  omitted-`max_tokens` requests serialize earlier on multi-sequence models. Accepted;
  it is honest and single-user.
