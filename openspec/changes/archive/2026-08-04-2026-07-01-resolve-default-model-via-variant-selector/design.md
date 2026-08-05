## Context

`variant_selector.select_variant(family, vram_budget)` was introduced in 3C and is called at model-load time when `INFERENCE_X_LOADED_MODELS` contains a family name. `INFERENCE_X_DEFAULT_MODEL` is a separate env var resolved independently — it does a direct name lookup against the loaded model registry and does not call `select_variant`. This creates a split: family names work in the loaded-model list but not in the default-model path.

The fix is a single callsite change: when the default-model value is present in the family registry (i.e. it matches a `ModelEntry.family` value), resolve it through `select_variant` before the lookup proceeds. If it is not a family name, leave the existing path unchanged.

## Goals / Non-Goals

**Goals:**
- `INFERENCE_X_DEFAULT_MODEL` accepts a family name and resolves to the best-fit variant at startup.
- Concrete model names continue to work with zero behavioral change.
- The resolved name is logged at startup (INFO level).
- A clear error is raised (not a silent fallback) if the family name is present but no variant fits the current VRAM tier.

**Non-Goals:**
- Do not change `variant_selector.select_variant()` internals.
- Do not change `INFERENCE_X_LOADED_MODELS` resolution logic.
- Do not add per-request default-model resolution — this is operator config, resolved once at startup.
- Do not add new API fields or change the request schema.

## Decisions

### Resolution happens at startup, not per-request

`INFERENCE_X_DEFAULT_MODEL` is operator config. Resolving it once at startup is consistent with how tier resolution and model loading already work. Per-request resolution would add a VRAM-check branch to the hot path for no benefit, since the loaded model set is fixed at startup.

Alternative considered: resolve per-request based on live VRAM availability. Rejected — this turns a static operator setting into a dynamic routing decision, which belongs in the admission controller and variant selector, not in the default-model lookup.

### Family-name detection uses the model registry, not a separate flag

If the value of `INFERENCE_X_DEFAULT_MODEL` matches any `ModelEntry.family` in the loaded registry, treat it as a family name and invoke `select_variant`. Otherwise treat it as a concrete name.

Alternative considered: require a separate `INFERENCE_X_DEFAULT_MODEL_IS_FAMILY=true` flag. Rejected — adds operator burden for a case that can be detected unambiguously from the registry.

### Missing-family-fit is a hard startup error

If the family is recognized but `select_variant` returns no fitting variant (VRAM too tight), raise a startup error with a clear message listing the family, the available variants, and the current tier budget. Do not fall back silently to the first variant.

Alternative considered: log a warning and fall back to the first variant. Rejected — silent fallback violates the "no VRAM surprises" contract established in Phase 1–3.

## Risks / Trade-offs

- Operators who previously used a family name as `INFERENCE_X_DEFAULT_MODEL` and got a silent pass-through will now get either a correct resolution or a hard error. This is a behavior change but a correct one — document in release notes.
- If the VRAM tier at startup is lower than expected (e.g. another process consumed VRAM), the selected variant may be lower-precision than the operator intended. The startup log makes this observable.
- The detection logic (family name vs concrete name) must be consistent with how 3C registers families — verify the family field is always populated for models that opt into variant grouping.

## Migration Plan

1. Identify the callsite where `INFERENCE_X_DEFAULT_MODEL` is resolved (likely `AppSettings` or model registry init).
2. After the loaded-model registry is populated, check if the default-model value matches any `ModelEntry.family`.
3. If yes, call `select_variant(family, current_tier.vram_budget_gb)` and replace the default value with the returned variant name.
4. Log at INFO: `Default model resolved: {family} → {variant} (tier: {tier_name})`.
5. If `select_variant` returns `None`, raise `RuntimeError` with a descriptive message.
6. Add unit tests: family resolved correctly, concrete name unchanged, missing family raises error.
7. Update `README.md`, `docs/DECISIONS.md`, `docs/PHASES.md`, `article-notes.md`.
8. Run `uv run pytest tests/unit/ -v` — all tests must pass.
9. Live-verify: start the server with a family name as `INFERENCE_X_DEFAULT_MODEL`, confirm the startup log shows the resolved variant, send one request without specifying a model, confirm it routes to the resolved variant.