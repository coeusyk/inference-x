# Platform Specification Delta — verify-benchmark-suite-identity

## ADDED Requirements

### Requirement: Benchmark suite identity is a pinned content hash
The platform SHALL identify a benchmark prompt suite by the SHA-256 digest of the
canonical JSON encoding of the suite's **parsed in-memory prompt collection**, and
SHALL NOT derive suite identity from the suite file's raw bytes, encoding,
whitespace, indentation, line endings, JSON object key order, or serialization
formatting. The digest SHALL use exactly:

`sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()`

where `prompts` is `data["prompts"]` after JSON parsing. The stored `suite_version`
key SHALL be excluded from the hashed content. The digest SHALL be stored and
compared as bare hexadecimal (no algorithm-tag prefix). This requirement records
DEC-054.

#### Scenario: Shipped suite digest matches characterization proof
- **WHEN** the shared canonicalizer hashes the prompts of the shipped standard suite
- **THEN** the digest equals
  `b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4`
- **AND** no rewrite of the suite file is required

#### Scenario: Formatting is excluded from identity
- **WHEN** two suite files parse to the same prompt collection but differ only in
  whitespace, indentation, line endings, JSON object key order, or serialization
  formatting
- **THEN** they produce the same `suite_version` digest

#### Scenario: Prompt values and order participate in identity
- **WHEN** any prompt `label`, prompt `text`, or prompt ordering changes
- **THEN** the digest differs from the prior suite identity

#### Scenario: Suite version key is not hashed
- **WHEN** only the `suite_version` field in the suite file differs from a prior copy
  while `prompts` are unchanged
- **THEN** the computed digest is unchanged
- **AND** load-time verification still compares that digest to the stored field

### Requirement: One shared canonicalizer owns suite identity
The platform SHALL provide exactly one canonicalizer implementation for suite
identity. Every producer and consumer of `suite_version` digests — including suite
load verification and the regeneration command — SHALL call that implementation.
The platform SHALL NOT duplicate hashing logic inline in the runner, storage layer,
scripts, or tests-as-production-paths.

#### Scenario: Load verification and regeneration share one implementation
- **WHEN** `make suite-version` (or its script entry point) computes a digest and the
  suite loader verifies a suite
- **THEN** both call the same canonicalizer module
- **AND** neither embeds a second `json.dumps` / `sha256` canonicalization path

### Requirement: Suite load fails loudly on identity errors
The platform SHALL, when loading a benchmark suite, compute the pinned digest of the
parsed prompts and compare it to the suite's stored `suite_version`. On mismatch or
on a missing `suite_version` key, the platform SHALL raise an actionable error that
names `make suite-version`, and SHALL NOT proceed with an unverified suite. This
fail-loud posture applies to the benchmark harness and SHALL NOT be reinterpreted as
admission fail-open (DEC-047 §4).

#### Scenario: Mismatch between stored and computed suite version
- **WHEN** a suite file's stored `suite_version` disagrees with the canonicalizer
  digest of its prompts
- **THEN** suite load raises
- **AND** the error message names `make suite-version`

#### Scenario: Missing suite version key
- **WHEN** a suite file omits `suite_version`
- **THEN** suite load raises an actionable error naming `make suite-version`
- **AND** the failure is not an unadorned `KeyError`

### Requirement: Storage filters by suite version before latest-per-model selection
The platform SHALL filter benchmark results by the expected `suite_version` before
selecting the latest result per model. Results whose `suite_version` does not match
the expected suite SHALL NOT be eligible for latest-per-model selection and SHALL
NOT be ranked against the current suite. Filtering SHALL live in the storage
selection path, not in the advisor ranking module. This requirement records the
consumption side of DEC-055.

#### Scenario: Mixed-suite comparisons are impossible
- **WHEN** the result store contains recent results for the same model under two
  different suite versions and the current expected suite is one of them
- **THEN** latest-per-model selection returns only results for the expected suite
- **AND** the advisor is never handed a mixed-suite set from this selection path

#### Scenario: Matching suite results still select latest per model
- **WHEN** multiple results share the expected `suite_version` for one model
- **THEN** selection returns the most recent among those matching results

### Requirement: Empty store and suite-version mismatch are distinct outcomes
The platform SHALL distinguish the operational outcome "no benchmark results exist"
from the outcome "benchmark results exist but none match the requested
`suite_version`". Selection MUST NOT silently present a suite mismatch as an empty
store without a distinguishable signal.

#### Scenario: Empty store
- **WHEN** the results directory contains no benchmark result files
- **THEN** selection reports an empty-store outcome
- **AND** that outcome is distinguishable from a suite mismatch

#### Scenario: Suite mismatch with existing results
- **WHEN** benchmark result files exist but none carry the expected `suite_version`
- **THEN** selection reports a suite-mismatch outcome
- **AND** that outcome is distinguishable from an empty store
- **AND** no mismatched results are returned for ranking

### Requirement: Benchmark results are immutable and append-only
The platform SHALL treat stored benchmark result files as immutable and append-only
with respect to suite identity and comparability. Incomparable or mismatched results
SHALL be excluded by selection, and SHALL NOT be deleted or rewritten to satisfy
comparability. Rollback of this change SHALL NOT require rewriting historical
benchmark files.

#### Scenario: Exclusion does not delete history
- **WHEN** results from a non-matching suite version are present
- **THEN** they remain on disk after suite-aware selection
- **AND** selection simply omits them from the returned set

#### Scenario: Rollback needs no historical rewrite
- **WHEN** this change is reverted
- **THEN** existing result JSON files remain valid inputs without migration
- **AND** no rewrite of historical files was required by this change

### Requirement: Suite version is necessary but not sufficient for comparability
The platform SHALL treat `suite_version` as identity of the benchmark input only. The
platform SHALL NOT treat equal `suite_version` values alone as proof that two results
are fully comparable across hardware, runtime, or other execution dimensions.
This requirement records DEC-055.

#### Scenario: Equal suite version does not claim full provenance
- **WHEN** two stored results share the same `suite_version`
- **THEN** the platform may use that agreement as a necessary comparability gate
- **AND** it does not treat that agreement as sufficient provenance by itself

### Requirement: Compatibility — score is a within-report ordinal
The platform SHALL treat advisor `score` as a within-report ordinal only. This change
SHALL NOT redefine `score` as an absolute metric, a portable quality scale across
reports, or a 0–100 quality score with meaning outside a single report.

#### Scenario: OS-5 does not redefine score semantics
- **WHEN** this change lands
- **THEN** advisor score calculation and interpretation are unchanged by OS-5 tasks
- **AND** no OS-5 code path documents or implements `score` as an absolute metric

### Requirement: Compatibility — viable is the only viability signal
The platform SHALL treat `viable` as the only viability indicator for advisor
recommendations. This change SHALL NOT introduce or rely on `score > 0` (or any
score threshold) as a viability proxy.

#### Scenario: OS-5 does not use score as viability
- **WHEN** OS-5 storage selection or suite verification runs
- **THEN** it does not interpret advisor `score` to decide viability
- **AND** existing `viable` semantics are left unchanged for OS-6

### Requirement: Compatibility — advisor owns scoring weights
The platform SHALL keep ownership of scoring weights in the advisor. Benchmark
collection and suite identity code SHALL NEVER embed, import, or assume advisor
scoring weights (including the frozen weight set `4/9`, `1/3`, `2/9`).

#### Scenario: Collection does not depend on advisor weights
- **WHEN** the benchmark runner records a result or the canonicalizer hashes a suite
- **THEN** neither path reads or embeds advisor scoring weights
- **AND** changing advisor weights cannot require changes to suite identity code

### Requirement: Compatibility — deprecated aliases remain through Phase A
The platform SHALL retain deprecated benchmark/advisor field aliases through Phase A.
This change SHALL NOT remove compatibility aliases. Removal of deprecated aliases
REQUIRES a future ADR and is out of scope for OS-5.

#### Scenario: OS-5 does not remove aliases
- **WHEN** this change lands
- **THEN** no deprecated benchmark JSON or HTTP field alias is removed
- **AND** no OS-5 task schedules alias removal without a future ADR

## ADDED Requirements

### Requirement: Reproducible benchmark results
The benchmark runner SHALL use a fixed, versioned prompt suite so results are
comparable across runs only when they share verified suite identity. The suite's
`suite_version` SHALL be the pinned content hash of the parsed prompt collection
(DEC-054), verified at load, and used as a necessary selection key before
latest-per-model consumption (DEC-055).

#### Scenario: Benchmark is run twice on the same hardware
- **WHEN** the same model is benchmarked twice with the standard prompt suite
- **THEN** results are stored separately with timestamps
- **AND** the advisor uses the most recent result per model among results that match
  the expected `suite_version`

#### Scenario: Unverified suite cannot produce results
- **WHEN** the prompt suite fails identity verification at load
- **THEN** the benchmark run does not proceed
- **AND** no new result file is written from that failed load
