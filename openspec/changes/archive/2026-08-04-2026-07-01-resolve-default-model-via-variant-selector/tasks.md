## 1. Core Resolution Logic

> Note on 1.3/1.5: the real `select_variant()` signature is `select_variant(family,
> registry, tier, available_vram_gib)` — `VramTier` has no `vram_budget_gb` attribute,
> and `select_variant` never returns `None`; it either returns a concrete name or raises
> `NoVariantFitsError` (already a `RuntimeError` subclass) with the family/variants/tier
> budget in its message. Implemented against the real signature/behavior below; the
> outcome (hard RuntimeError naming family, variants, and tier budget on no-fit) matches
> what 1.5 asks for.

- [x] 1.1 Identify the exact callsite where `INFERENCE_X_DEFAULT_MODEL` is resolved against the loaded model registry.
- [x] 1.2 After registry population, check if the value matches any `ModelEntry.family` in the loaded set.
- [x] 1.3 If a family match is found, call `variant_selector.select_variant(family, registry, tier, available_vram_gib)` and replace the setting value with the returned concrete name.
- [x] 1.4 Log at INFO level: `Default model resolved: {family} → {variant} (tier: {tier_name})`.
- [x] 1.5 `select_variant` raises `NoVariantFitsError` (a `RuntimeError`) for a recognized family with no fitting variant, with a message that includes the family name, available variants, and current tier budget — propagated unmodified, no `None`-check needed.

## 2. Tests

- [x] 2.1 Add `tests/unit/test_default_model_resolution.py` covering:
  - Family name resolves to highest-precision fitting variant.
  - Concrete model name passes through unchanged.
  - Family name with no fitting variant raises `RuntimeError` at startup.
  - Family name not present in registry is treated as a concrete name (existing path).

## 3. Documentation

- [x] 3.1 Add DEC-042 to `docs/DECISIONS.md`: documents family-name resolution for `INFERENCE_X_DEFAULT_MODEL`, the detection strategy, and the hard-error behavior.
- [x] 3.2 Update `docs/PHASES.md` with Phase 12 entry.
- [x] 3.3 Update `README.md`: note that `INFERENCE_X_DEFAULT_MODEL` accepts either a concrete model name or a registered family name; add a one-line example.
- [x] 3.4 Update `article-notes.md`: note that the default-model path now participates in VRAM-aware selection — this closes the last known gap in the Phase 3C scope boundary.

## 4. Validation

- [x] 4.1 Run `uv run pytest tests/unit/ -v` — all tests must pass (baseline: 419 + new tests).
- [x] 4.2 Start the server with `INFERENCE_X_DEFAULT_MODEL` set to a family name (`qwen2.5-7b`); confirm startup log shows `Default model resolved: qwen2.5-7b → qwen2.5-7b-awq (tier: 6gb)` or equivalent.
- [x] 4.3 Send one `POST /v1/chat/completions` without specifying a model; confirm the response's `model` field matches the resolved variant name.