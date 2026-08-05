# Platform Specification Delta — close-generate-stream-contract-gap

## ADDED Requirements

### Requirement: Engine streaming interface is declared as an async generator, not a coroutine returning one
`BaseEngine.generate_stream` SHALL be declared such that calling it directly produces an
async generator, matching how every implementation defines it and how every caller uses
it. It SHALL NOT be declared such that a type checker infers calling it returns a
coroutine that must be awaited before iteration.

#### Scenario: Declared type matches implementation type
- WHEN a concrete engine implements `generate_stream` as an async generator function
  (its body contains `yield`)
- THEN the abstract declaration's return type is satisfied without a type checker
  reporting an invalid-override error

#### Scenario: Callers iterate without awaiting
- WHEN a caller invokes `generate_stream(request)`
- THEN the returned object is immediately usable as an async generator (`__anext__`,
  `aclose`) without first being awaited

#### Scenario: Change is applied
- WHEN this change is applied
- THEN `mypy` reports no errors for `services/chat_service.py`
- AND `inference_x.services.chat_service` is removed from the DEC-048 mypy baseline
- AND no other module's baseline entry grows
- AND no runtime behavior, streaming event order, or `ChatStreamChunk` shape changes

### Requirement: Compatibility — abstract and implementation denote the same callable type
The abstract declaration and every implementation MUST denote the same callable type.
The platform SHALL enforce this as a Compatibility Invariant stronger than
"no invalid override": type agreement MUST hold for the current engine and for every
future `BaseEngine.generate_stream` implementation.

#### Scenario: Declaration and implementation share one callable type
- WHEN a concrete engine implements `generate_stream`
- THEN its callable type is the same type denoted by the abstract declaration
- AND a type checker does not report an invalid-override error for that method

### Requirement: Compatibility — no caller changes are permitted
No caller changes are permitted. The platform SHALL keep ChatService, EngineDriver,
middleware, routing, benchmarks, and tests valid without adaptation after the
declaration correction.

#### Scenario: Callers remain valid without adaptation
- WHEN the abstract declaration is corrected
- THEN ChatService, EngineDriver, middleware, routing, benchmarks, and tests remain
  valid without edits
- AND no caller is adapted to await or otherwise reinterpret `generate_stream`

### Requirement: Future generate_stream implementations type-check without suppression
Future implementations of `BaseEngine.generate_stream` MUST type-check without
requiring suppression. The platform SHALL treat a new implementation that needs a
mypy baseline entry or other suppression for this contract as a regression.

#### Scenario: New implementation needs no generate_stream suppression
- WHEN a future concrete engine implements `generate_stream` against the corrected
  declaration
- THEN that implementation type-checks without a mypy baseline entry, `# type: ignore`,
  or other suppression for the `generate_stream` contract
