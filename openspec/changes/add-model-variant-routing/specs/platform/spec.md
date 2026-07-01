# Platform Specification Delta — add-model-variant-routing

## ADDED Requirements

### Requirement: Model variant selection at load time
The system SHALL support grouping model configuration entries into named families
and, when an operator requests a family rather than a concrete model, SHALL select
the highest-precision variant that fits the available VRAM budget at load time.

#### Scenario: Multiple variants, one fits
- WHEN a model family has bf16, int8, and 4-bit quantized variants registered
- AND only the 4-bit variant's estimated weight size fits the available VRAM
  budget
- THEN the 4-bit variant is loaded
- AND the higher-precision variants are not loaded

#### Scenario: No variant fits
- WHEN no variant in a requested family fits the available VRAM budget
- THEN startup fails with an error naming the family and each variant's
  estimated size versus the available budget
- AND the process does not begin accepting requests

#### Scenario: Concrete model name bypasses selection
- WHEN `INFERENCE_X_LOADED_MODELS` names a concrete, registered model entry
  directly (not a family)
- THEN that exact entry is loaded, unaffected by variant selection

#### Scenario: Ungrouped model is unaffected
- WHEN a model entry has no `family` set
- THEN it loads exactly as it did before this change

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `TaskRouter` and `AdmissionController` behavior is unchanged — both
  continue to operate only on the concrete, already-resolved model name
- AND all existing unit tests continue to pass unchanged
