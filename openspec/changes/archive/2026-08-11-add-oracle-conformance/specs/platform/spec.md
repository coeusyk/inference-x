## ADDED Requirements

### Requirement: Gated oracle suite validates teacher-forced top-1 agreement vs transformers
The platform SHALL maintain a gated oracle conformance suite under `tests/oracle/` that
teacher-forces the registered `opt-125m` model (`facebook/opt-125m`) through the Inference-X
vLLM engine path and through raw `transformers`, and asserts top-1 token agreement per position on
a fixed plaintext prompt. The suite SHALL report floating-point near-ties by position rather than
counting them as silent agreements, and SHALL fail only on hard top-1 disagreements that are not
near-ties. The suite SHALL NOT run as part of the GPU-free automated `checks` job.

#### Scenario: Oracle asserts per-position top-1 agreement on opt-125m
- **WHEN** the gated oracle suite runs successfully against `opt-125m`
- **THEN** every prompt position is classified as exact top-1 agreement or a reported near-tie
- **AND** the report names each near-tie by position
- **AND** no hard top-1 disagreement remains unclassified as a near-tie

#### Scenario: Near-ties are reported, not silently passed
- **WHEN** a position's top-1 tokens disagree between Inference-X and `transformers`, but the
  logit margin on either side is within the suite's near-tie epsilon
- **THEN** the suite records that position as a near-tie in its report
- **AND** the position does not count toward the exact-agreement tally
- **AND** the position does not by itself fail the suite

#### Scenario: Hard mismatches fail the suite
- **WHEN** a position's top-1 tokens disagree and the disagreement is not a near-tie
- **THEN** the suite fails
- **AND** the report identifies the mismatched position

#### Scenario: Oracle is excluded from the GPU-free checks job
- **WHEN** the automated `checks` verification runs
- **THEN** it does not execute the gated oracle suite
- **AND** it still completes without a GPU

#### Scenario: Oracle requires an explicit GPU opt-in
- **WHEN** a contributor runs the oracle suite without the configured GPU opt-in environment flag
  or without a usable GPU
- **THEN** the suite skips with an explicit reason
- **AND** it does not start an engine or download model weights as a side effect of collection alone
