## Context

`config/vram_tiers.yaml` already declares `block_size`/`kv_cache_dtype`/
`max_num_seqs` per tier and `VramTier` (`utils/vram_tiers.py`) already parses them,
but nothing downstream ever reads `block_size`/`kv_cache_dtype` off a resolved tier,
and `max_num_seqs` only ever flows from `ModelEntry.max_num_seqs` (a per-model
override) straight into `VLLMEngine.__init__`'s `LLM(**kwargs)` — the tier's value
is never consulted as a ceiling. `_build_engine_pool` never calls
`settings.get_vram_tier()` at all; only `initialize_app()` does, and only to log it.
`AdmissionController` already resolves a tier for its context-length gate
(`_context_ceiling`) and its KV-token gate (`engine.kv_capacity_tokens * 0.9`), but
has no gate on concurrent sequence count.

## Goals / Non-Goals

**Goals:**
- Make every VRAM/concurrency knob already declared in `vram_tiers.yaml` actually
  reach the vLLM engine.
- Add the two knobs (`max_num_batched_tokens`, `enable_prefix_caching`) that matter
  most for 6GB-tier stability but aren't represented anywhere yet.
- Give `max_num_seqs` saturation an explicit, observable failure mode (429) instead
  of invisible internal queuing inside vLLM's scheduler.
- Keep every change backward-compatible with existing `vram_tiers.yaml`/
  `models.yaml` files that don't set the new fields.

**Non-Goals:**
- No free-form per-request API surface for any of these knobs — they are operator
  config (tier + optional per-model override), never `ChatCompletionRequest` fields.
- No second scheduler layered on top of vLLM's own — the new admission gate only
  rejects/observes, it never reorders or holds requests itself.
- No change to `kv_cache_dtype` beyond passing the tier's declared value through —
  no automatic fp8-KV enablement or accuracy trade-off decisions in this change.

## Decisions

**Knobs live in `vram_tiers.yaml` (tier ceiling) + `models.yaml` (optional
per-model override) — no new config file.** This matches the existing
`max_model_len_cap`/`ModelEntry.max_model_len` and `tier.max_num_seqs`/
`ModelEntry.max_num_seqs` pattern exactly. A third config file
(`engine_knobs.yaml`) would fragment resolution logic across three files instead
of two for no benefit, since `vram_tiers.yaml` already documents these fields as
"ready to wire in without a config-schema change".

**Resolution is `min(model override or tier value, tier value)`, matching
`AdmissionController._context_ceiling`'s existing composition.** A model can only
ask for a *tighter* budget than its tier allows (e.g. a Mamba/hybrid model
needing fewer concurrent sequences), never a looser one — the tier ceiling is a
hardware-derived safety bound, not a suggestion.

**`enable_prefix_caching` is tier-only, no per-model override.** It's a single
engine-startup flag, and this codebase runs one model per engine process — there's
no meaningful "this model wants prefix caching, that one doesn't" distinction
within one tier. Adding a per-model override here would be unused surface area.

**Default values for the 6GB tier:** `max_num_batched_tokens: 2048` (bounds a
transient prefill-burst spike to roughly one full-context prefill at a time — the
knob most directly missing today, since KV steady-state usage is already bounded by
`max_num_seqs`/`max_model_len_cap` but a burst of concurrent long prompts hitting
the scheduler in the same step is not), `enable_prefix_caching: false` (prefix
caching keeps extra KV blocks alive longer than strictly needed — works against a
tier that's already tight on KV budget; enabled on 12gb/24gb where there's
headroom). `block_size`/`kv_cache_dtype` keep their existing tier defaults (16 /
`auto`) — this change only makes those existing values actually reach `LLM(...)`.

**The new sequence-count gate has no clamp path, unlike the token gate.** The
existing KV-token gate can clamp `interactive` requests down to a smaller
`max_tokens` that still fits. Sequence-count saturation has no equivalent: a
request either gets a slot or it doesn't, there's no partial slot. So both
priorities get `EngineSaturatedError` (429) when the in-flight count for a model
is at or above its resolved `max_num_seqs` — this is a deliberate asymmetry with
the token gate, called out here so it isn't mistaken for an oversight.

## Risks / Trade-offs

- vLLM kwarg names/accepted ranges for `max_num_batched_tokens`/
  `enable_prefix_caching` can drift across versions — this codebase already hit
  this exact class of problem in `_log_kv_cache_stats`'s `cache_config`
  introspection (vLLM 0.22.1's V1 engine moved `cache_config` under
  `vllm_config`). Needs the same defensive, exception-tolerant handling, and the
  vLLM version this was verified against recorded in the change's DEC entry.
- A too-conservative `max_num_batched_tokens` degrades throughput (more, smaller
  scheduler steps) rather than crashing — nothing in a unit test suite catches
  "correct but slow"; a live throughput comparison is out of scope here
  (observability/benchmarking is explicitly excluded from this phase).
- If KV-token pressure and sequence-count pressure trip on the same request,
  `admit()` must raise exactly one clear, gate-attributed error — a caller
  shouldn't see two stacked/ambiguous rejection reasons for one request.
