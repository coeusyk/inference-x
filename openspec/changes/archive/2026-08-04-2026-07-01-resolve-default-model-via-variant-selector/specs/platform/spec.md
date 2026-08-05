# Platform Specification Delta — resolve-default-model-via-variant-selector

## ADDED Requirements

### Requirement: Default model resolves family names via variant selection
The system SHALL resolve `INFERENCE_X_DEFAULT_MODEL` through
`variant_selector.select_variant()` when its value matches a registered model
family, so the highest-precision variant that fits the current VRAM tier is
selected at startup. A value that is already a concrete registered model name
SHALL continue to pass through unchanged.

#### Scenario: Family name resolves to the best-fit variant at startup
- WHEN `INFERENCE_X_DEFAULT_MODEL` matches a `ModelEntry.family` in the loaded
  registry and a VRAM tier is resolved
- THEN the value is resolved via `select_variant()` to the highest-precision
  variant that fits the current tier's available VRAM
- AND that resolved variant name is used to construct the default-model policy

#### Scenario: Concrete model names pass through unchanged
- WHEN `INFERENCE_X_DEFAULT_MODEL` is already an exact registered model name
- THEN the value is used unchanged
- AND no call to `select_variant()` is made

### Requirement: Default model resolution surfaces its outcome, not silence
The system SHALL make the outcome of default-model resolution observable and
SHALL NOT silently fall back to an arbitrary variant when a recognized family
has no fitting variant for the current VRAM tier.

#### Scenario: Resolution is logged at startup
- WHEN a family name is resolved to a concrete variant
- THEN the system logs, at INFO level, the family name, the resolved variant
  name, and the current tier name

#### Scenario: Missing-family-fit is a hard startup error
- WHEN `INFERENCE_X_DEFAULT_MODEL` matches a registered family but no variant
  in that family fits the current VRAM tier's available budget
- THEN startup raises an error naming the family, the available variants, and
  the current tier's VRAM budget
- AND the system does not silently start with a different, unrequested variant
