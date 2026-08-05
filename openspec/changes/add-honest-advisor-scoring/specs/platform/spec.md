# Platform Specification Delta — add-honest-advisor-scoring

## ADDED Requirements

### Requirement: Advisor score uses measured quantities only
The platform SHALL compute the advisor `score` from measured throughput, TTFT, and
VRAM components only, using the exact weights throughput `4/9`, TTFT `1/3`, and
VRAM `2/9`. The platform SHALL NOT include a constant quantization placeholder or
any other non-measured component in the score. This requirement records DEC-056.
The platform SHALL treat score as follows: score is a within-report ordinal used
only to rank viable models produced from the same benchmark suite.

#### Scenario: Constant quant score is removed
- **WHEN** the advisor ranks one or more viable models
- **THEN** the score formula does not add a constant quantization term
- **AND** the three component weights are exactly `4/9`, `1/3`, and `2/9`

#### Scenario: Viable ranking order is preserved across the weight change
- **WHEN** a fixed set of viable models is ranked under the previous measured-only
  weighted sum and under the exact frozen weights
- **THEN** the relative order among those viable models is unchanged

#### Scenario: Score monotonicity with one component varied
- **WHEN** two viable models differ in exactly one measured component while the
  other components are held constant
- **THEN** the model with the better value of that component ranks above the other
- **AND** this holds independently for throughput, TTFT, and VRAM components

### Requirement: Compatibility — score is a within-report ordinal
The platform SHALL define score as follows: score is a within-report ordinal used
only to rank viable models produced from the same benchmark suite. The platform
SHALL NOT treat score as portable across benchmark suites, portable across
hardware, portable across future weighting changes, or an absolute quality metric.

#### Scenario: Single-result report exposes relative TTFT floor
- **WHEN** the advisor ranks a report containing exactly one viable result
- **THEN** that result's relative TTFT component is zero against the report max
- **AND** score is a within-report ordinal used only to rank viable models produced
  from the same benchmark suite

#### Scenario: Adding a model can change other scores
- **WHEN** a second model is added to a previously single-model report and the
  advisor re-ranks
- **THEN** at least one previously reported score may change
- **AND** that change does not imply the first model became objectively worse
  outside the report

#### Scenario: Score is not portable across suites, hardware, or weights
- **WHEN** a client reads an advisor `score`
- **THEN** the platform does not claim the score is comparable across benchmark
  suites, across hardware profiles, or across future weighting changes
- **AND** the platform does not claim the score is an absolute quality metric

### Requirement: Compatibility — viable is the only viability signal
The platform SHALL treat `viable` as the only viability indicator for advisor
recommendations. The platform SHALL NOT use `score > 0`, or any other score
threshold, as a viability proxy. Every consumer of advisor output SHALL use
`viable`; no consumer SHALL derive viability from score.

#### Scenario: Viable-worst and non-viable both may score zero
- **WHEN** a VRAM-gated model and a viable model with zero relative measured
  contribution both appear in one report
- **THEN** both may have `score` equal to `0.0`
- **AND** they remain distinguishable by the `viable` field alone

#### Scenario: Score threshold is not viability
- **WHEN** a client inspects advisor output
- **THEN** non-viability is indicated only by `viable == false`
- **AND** a zero score alone does not imply non-viability

#### Scenario: Consumers use viable, never score, for viability
- **WHEN** `/v1/benchmark/advise` or `make advise` presents recommendations
- **THEN** viability is taken from the `viable` field
- **AND** no consumer derives viability from score ordering or thresholds

### Requirement: Compatibility — advisor owns scoring weights
The platform SHALL keep ownership of scoring weights in the advisor. The advisor
owns the scoring weights `4/9`, `1/3`, and `2/9`. The runner must never embed them.
Storage must never embed them. Schemas must never embed them. This is a repository
ownership rule: suite identity and storage selection also SHALL NEVER embed,
import, or assume those weights.

#### Scenario: Collection does not depend on advisor weights
- **WHEN** the benchmark runner records a result
- **THEN** it does not read or embed advisor scoring weights
- **AND** changing advisor weights does not require changes to suite identity or
  storage filtering

#### Scenario: Schemas and storage do not embed weights
- **WHEN** benchmark schemas and storage selection code are inspected
- **THEN** neither embeds the scoring weights `4/9`, `1/3`, or `2/9`
- **AND** weight ownership remains exclusively in the advisor

### Requirement: Canonical VRAM metric is device occupancy
The platform SHALL persist and expose the benchmark VRAM measurement under the
canonical field name `vram_device_occupied_gib`. The value SHALL remain
whole-device occupancy computed as `total − min(free)` at sampled free-memory
boundaries, including other GPU users. The platform SHALL NOT redefine the
measurement as a per-model footprint. This requirement records DEC-057.

#### Scenario: Canonical field name on new results
- **WHEN** a new benchmark result is written
- **THEN** it carries `vram_device_occupied_gib`
- **AND** the numeric value equals the existing device-occupancy computation

#### Scenario: HardwareProfile names unchanged
- **WHEN** this change lands
- **THEN** `HardwareProfile.vram_total_gb` and `HardwareProfile.vram_free_gb`
  retain those names
- **AND** no HardwareProfile rename is required for OS-6

### Requirement: Compatibility — deprecated aliases remain through Phase A
The platform SHALL accept and expose the following deprecated aliases through
Phase A:

- `peak_vram_delta_gb` as an alias of `vram_device_occupied_gib` on stored
  benchmark results
- `vram_gb` as an alias of `vram_device_occupied_gib` on advisor results

Removal of either alias REQUIRES a future ADR and is out of scope for this
change. Canonical fields always take precedence over deprecated aliases during
deserialization. If both are present with conflicting values, the canonical field
wins. This precedence rule is permanent unless superseded by a future ADR.

#### Scenario: Historical results load via alias
- **WHEN** a stored result JSON carries `peak_vram_delta_gb` and omits the
  canonical key
- **THEN** the platform loads it as `vram_device_occupied_gib`
- **AND** the file is not rewritten

#### Scenario: Advisor wire alias retained
- **WHEN** an advisor result is serialized
- **THEN** `vram_device_occupied_gib` is the canonical field
- **AND** `vram_gb` remains available as a deprecated alias through Phase A
- **AND** the serialized payload still includes `vram_gb`

#### Scenario: Canonical-only round-trip
- **WHEN** input supplies only `vram_device_occupied_gib`
- **THEN** deserialization succeeds with that value

#### Scenario: Alias-only round-trip
- **WHEN** input supplies only `peak_vram_delta_gb` (or advisor `vram_gb`)
- **THEN** deserialization succeeds and populates the canonical field

#### Scenario: Canonical and alias present without conflict
- **WHEN** input supplies both the canonical field and a deprecated alias with
  equal numeric values
- **THEN** deserialization succeeds with that value

#### Scenario: Conflicting dual values — canonical wins
- **WHEN** input supplies both the canonical field and a deprecated alias with
  unequal numeric values
- **THEN** the canonical field value is used
- **AND** the alias value is ignored for the resolved field

### Requirement: No benchmark migration under OS-6
The platform SHALL NOT rewrite historical benchmark result JSON as part of this
change. Historical benchmark JSON remains readable without migration. New
benchmark JSON uses the canonical field `vram_device_occupied_gib`. Deprecated
aliases remain readable throughout Phase A. Rollback SHALL require only reverting
code, never rewriting stored results.

#### Scenario: Results directory remains append-only for this change
- **WHEN** OS-6 lands
- **THEN** existing files under `benchmarks/results/` are unmodified by the change
- **AND** new runs may append new files that use the canonical field
- **AND** historical files remain readable via deprecated aliases

## MODIFIED Requirements

### Requirement: Hardware-aware model recommendation
The system SHALL measure model performance on the operator's hardware and produce a
ranked recommendation with plain-language reasoning. Ranking SHALL use only measured
score components under the exact frozen weights, and SHALL expose viability solely
through `viable`. score is a within-report ordinal used only to rank viable models
produced from the same benchmark suite.

#### Scenario: Advisor is run after benchmarking
- **WHEN** `make advise` is run after at least one benchmark result exists for the
  current suite
- **THEN** the advisor produces a ranked list of models
- **AND** each entry includes: throughput (tok/s), TTFT (ms), device VRAM occupancy
  (GiB), a score that is a within-report ordinal used only to rank viable models
  produced from the same benchmark suite, and a one-line recommendation string
- **AND** models that exceed available VRAM are flagged as not viable via `viable`
  rather than by score threshold alone
