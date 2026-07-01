## Context

`config/models.yaml` entries are flat and independent; `ModelRegistry` indexes them
by `name` alone. `INFERENCE_X_LOADED_MODELS` names concrete entries, and
`_build_engine_pool` loads exactly those. `utils/vllm_pool_config.py` already has a
quant-aware bytes-per-param table (`_QUANT_BYTES_PER_PARAM`: bf16=2.0, int8/fp8=1.0,
awq/gptq/int4=0.55) and a weight-size estimator built on it (Phase 1,
DEC-037/DEC-036) — used today only to size a single already-chosen model's memory
footprint, never to choose *between* multiple candidates.

DEC-038 explicitly named `precision` as a request field that was "deliberately not
added — it's meaningless without variant sets... which don't exist yet; adding an
inert field would be dead API surface." This change adds the variant-set concept
that field would need; it does not itself add a `precision` request field or any
per-request variant selection — that remains out of scope until this groundwork is
proven.

## Goals / Non-Goals

**Goals:**
- Let one logical model be declared as multiple config entries at different
  quantizations, grouped by an explicit `family` key.
- Automatically pick the highest-precision variant that fits the current VRAM
  tier's budget at load time, reusing the existing weight estimator as the sole
  source of truth for both "how big is this variant" and "which is higher
  precision than which."
- Fail fast and loudly if no variant fits, matching the existing
  `initialize_app()` posture (engines not healthy after startup → hard error, no
  silent partial start).
- Stay fully backward compatible: every existing `models.yaml` entry and every
  existing `INFERENCE_X_LOADED_MODELS` value keeps working unchanged.

**Non-Goals:**
- No per-request variant/precision selection, no `precision` request field, no
  changes to `TaskRouter` or `AdmissionController` — selection happens once, at
  load time, not per request.
- No new precision-ranking field or config — order is derived entirely from
  `_QUANT_BYTES_PER_PARAM`, which already exists and is already the source of
  truth for weight-size estimation.
- No tie-breaking mechanism for two variants at the same precision rank — resolved
  by config order, documented as a known sharp edge rather than engineered away.

## Decisions

**Variants are grouped by an additive `family` field, not a new top-level YAML
structure.** A new structure (e.g. nesting variants under a family key) would
require rewriting every existing `models.yaml` entry's shape. An optional flat
field keeps every current entry valid as-is — the "family of one" case (no
`family` set) is not a special case in the code, it is what
`ModelRegistry.variants()` naturally returns for any ungrouped name.

**Precision order is derived from `_QUANT_BYTES_PER_PARAM`, never stored as a
separate rank.** The alternative — an explicit `precision_rank: int` field per
entry — creates two sources of truth that can silently disagree (e.g. someone adds
a new quant scheme to the bytes-per-param table but forgets to update every
entry's rank). Sorting variants by `-bytes_per_param(m.quantization)` is exactly
"prefer bf16, then int8/fp8, then 4-bit," which is already the intended order and
requires no new data.

**Selection is load-time, not per-request.** This codebase's existing model is:
`INFERENCE_X_LOADED_MODELS` fixes which engines are resident for the life of the
process; `TaskRouter` only ever picks among already-loaded engines. Variant
selection plugs into that same seam — resolving a family name to a concrete
variant *before* engine construction — rather than adding a new per-request
decision point that `AdmissionController` and `TaskRouter` would both need to
understand. This keeps the change small and consistent with how the rest of the
system already works.

**No-fit is a hard startup failure, not a silent fallback.** Silently loading
"whatever fits, even the smallest/worst variant available" without telling the
operator would hide a capacity problem until first use. `NoVariantFitsError`
(naming the family and each variant's estimated size vs. available budget) is
raised at pool-build time, propagating to the same `initialize_app()` failure path
that already refuses to accept requests when the pool isn't healthy.

**`INFERENCE_X_LOADED_MODELS` keeps both meanings.** A family name resolves
through `select_variant`; a concrete variant name (matching a `ModelEntry.name`
exactly) bypasses the selector and loads exactly that entry, unchanged from
today. Distinguishing the two is a simple lookup: if the given name matches a
registered `ModelEntry.name` directly, treat it as concrete; only fall back to
family resolution if it doesn't (this also naturally handles a family name that
happens to collide with no entry's exact name).

## Risks / Trade-offs

- Two variants in the same family at equal precision rank (e.g. two different
  int8 configs) resolve by `models.yaml` insertion order — implicit and easy to
  get wrong; documented here and in code comments rather than solved with a
  tie-breaker field, per the goal of not over-engineering this before it's needed.
- The reused weight estimator is unit-tested but not live-verified above 6GB
  (per DEC-037/DEC-038) — a wrong estimate could select a variant that then fails
  to load with an actual OOM at engine-construction time. Existing
  `_check_vram_budget`/`_map_vllm_init_error` guards still turn this into a clear,
  attributed hard failure rather than silent corruption, but it is a real
  "wrong choice, loud failure" risk rather than "no failure at all."
- Overloading `INFERENCE_X_LOADED_MODELS` with two meanings (family vs. concrete
  name) is a small ergonomic risk: an operator renaming a model to also match an
  unrelated family name could change resolution behavior unexpectedly. Mitigated
  by checking for an exact concrete-name match first, always.
