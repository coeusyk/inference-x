# Platform Specification Delta — add-benchmark-advisor

## MODIFIED Requirements

**Requirement: Hardware-aware model recommendation**
The system SHALL measure model performance on the operator's hardware and produce a
ranked recommendation with plain-language reasoning.

Scenario: Advisor is run after benchmarking
- WHEN `make advise` is run after at least one benchmark result exists
- THEN the advisor produces a ranked list of models
- AND each entry includes: throughput (tok/s), TTFT (ms), VRAM usage (GB),
  and a one-line recommendation string
- AND models that exceed available VRAM are flagged as not viable

**Requirement: Reproducible benchmark results**
The benchmark runner SHALL use a fixed, versioned prompt suite so results
are comparable across runs and hardware configurations.

Scenario: Benchmark is run twice on the same hardware
- WHEN the same model is benchmarked twice with the standard prompt suite
- THEN results are stored separately with timestamps
- AND the advisor uses the most recent result per model

**Requirement: Non-invasive integration**
The benchmark module SHALL NOT change existing API contracts or require
modifications to Phase 1–5 route handlers.

Scenario: Benchmark module is added
- WHEN the benchmark module is introduced
- THEN all existing tests continue to pass unchanged
- AND the benchmark API endpoints are additive (/v1/benchmark/*)