## Why

`config/vram_tiers.yaml` already declares `block_size`, `kv_cache_dtype`, and
`max_num_seqs` per tier (since DEC-037), with a comment noting they are "not yet
consumed by the engine (surfacing them is Phase 2 work)". `ModelEntry.max_num_seqs`
already exists as a per-model override and is already passed to `LLM(**kwargs)` in
`VLLMEngine.__init__`. But nothing resolves the tier against the model to produce
one effective value, `_build_engine_pool` (`api/deps.py`) never resolves a VRAM tier
at all (only `initialize_app()` does, purely for a log line), and `block_size`/
`kv_cache_dtype` never reach `LLM(...)` despite being declared. Two more knobs that
matter for VRAM stability — `max_num_batched_tokens` (bounds the transient
compute/activation spike during prefill-heavy bursts, distinct from steady-state KV
usage) and `enable_prefix_caching` — aren't represented anywhere yet.

Separately, `AdmissionController`'s KV-pressure gate only tracks token budget. It has
no visibility into `max_num_seqs` — if a tier caps concurrent sequences at 4 and a
5th request arrives, vLLM's internal scheduler just queues it silently; the operator
gets a latency spike with no explicit signal, where a 429 (matching the existing KV-
saturation behavior) would be more useful, especially for batch-tier requests.

## What Changes

- Extend `VramTier` (`utils/vram_tiers.py`) and `config/vram_tiers.yaml` with
  `max_num_batched_tokens` and `enable_prefix_caching` per tier, `.get()`-defaulted
  for backward compatibility with existing tier files.
- Add an optional per-model `ModelEntry.max_num_batched_tokens` override
  (`schemas/model.py`), same shape as the existing `max_num_seqs` override.
- New `apply_tier_knobs(model_config, tier)` helper (`utils/vllm_pool_config.py`)
  resolving one effective value per knob via `min(model override or tier value, tier
  value)` — the same composition pattern `AdmissionController._context_ceiling`
  already uses for `max_model_len`.
- `VLLMEngine.__init__` passes `max_num_batched_tokens`, `enable_prefix_caching`,
  `block_size`, and `kv_cache_dtype` into `LLM(**kwargs)` when present (today only
  `max_num_seqs` makes this trip).
- `_build_engine_pool` (`api/deps.py`) resolves the VRAM tier (same fail-open
  try/except pattern already used twice in this file) and calls `apply_tier_knobs`
  per model before constructing each `VLLMEngine`.
- `AdmissionController` gains a second, binary in-flight-sequence-count gate
  alongside the existing token-based KV gate, seeded with the effective
  `max_num_seqs` for the routed model. Unlike the token gate, there is no clamp
  path — hitting the ceiling is always `EngineSaturatedError` (429), for both
  `interactive` and `batch` priority, since occupying a sequence slot isn't
  something a smaller `max_tokens` can fix.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `platform`: engine-level VRAM/concurrency knobs declared in config are actually
  applied at engine construction time; admission control also rejects requests that
  would exceed the resolved sequence-concurrency ceiling.

## Impact

- `config/vram_tiers.yaml`, `src/inference_x/utils/vram_tiers.py`: two new tier
  fields.
- `src/inference_x/schemas/model.py`: one new optional `ModelEntry` field.
- `src/inference_x/utils/vllm_pool_config.py`: new `apply_tier_knobs` helper.
- `src/inference_x/engines/vllm_engine.py`: four more kwargs passed to `LLM(...)`.
- `src/inference_x/api/deps.py`: `_build_engine_pool` resolves the tier and applies
  knobs before engine construction.
- `src/inference_x/routing/admission.py`: new `_InFlightSeqTracker`, wired into
  `admit()`/`release()`.
- No change to `ChatCompletionRequest` or any API contract — these are operator
  config knobs, not per-request parameters.
