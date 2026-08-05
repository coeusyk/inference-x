## Why

`INFERENCE_X_DEFAULT_MODEL` is resolved by a direct model-name lookup, bypassing `variant_selector.select_variant()`. Operators who configure a family name as the default (e.g. `qwen2.5-7b`) currently get a lookup failure or fall through to the raw name — the variant selector introduced in 3C (70ca241) is never consulted for this path. This means the VRAM-budget-aware selection built in Phase 3C is silently skipped for the most common operator touchpoint.

The scope boundary was noted in 3C and documented here for follow-up.

## What Changes

- Route `INFERENCE_X_DEFAULT_MODEL` through `variant_selector.select_variant()` when the value matches a registered family name, so the highest-precision variant that fits the current VRAM tier is resolved at startup.
- If the value is already a concrete variant name (existing behavior), pass it through unchanged — no regression to current operators.
- Log the resolved variant name at startup so operators can confirm which variant was selected.
- Add DEC-042 recording this decision and the resolution behavior.

## Capabilities

### Modified Capabilities
- `platform`: `INFERENCE_X_DEFAULT_MODEL` now accepts a family name in addition to a concrete model name. Resolution uses `variant_selector.select_variant()` against the current VRAM tier at startup.

## Impact

- Config: `INFERENCE_X_DEFAULT_MODEL` gains family-name support; concrete names continue to work unchanged.
- Startup: `AppSettings` or the model registry resolves the default at load time and logs the selected variant.
- Routing: no changes to `routing/variant_selector.py` logic — only the callsite in settings/registry startup is new.
- Tests: new unit tests for family-name resolution, concrete-name passthrough, and missing-family error behavior.
- Docs: `README.md`, `docs/DECISIONS.md` (DEC-042), `docs/PHASES.md` (Phase 12), and `article-notes.md` updated.