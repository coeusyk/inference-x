# OS-5 — Verify benchmark suite identity

## Why

`suite_version` is documented as a content hash of the prompt suite, but nothing
computes or verifies it. The stored literal in `benchmarks/prompts/standard.json`
already equals the pinned SHA-256 of the prompt collection; what is missing is
verification and a consumer that refuses mixed-suite comparisons. Without both,
suite identity remains decorative and advisor rankings can silently mix
incomparable results.

This change is **verification, not migration**. The frozen architecture proved the
shipped digest already matches; historical results stay comparable and are never
rewritten.

## What Changes

- Pin suite identity to a single shared canonicalizer over the **parsed in-memory
  prompt collection** (DEC-054).
- Fail loudly at suite load when `suite_version` is missing or disagrees with the
  computed digest; name `make suite-version` in the error.
- Add `make suite-version` / `scripts/suite_version.py` that call the **same**
  canonicalizer — no duplicate hash implementations.
- Filter stored results by expected `suite_version` in `storage.py` **before**
  latest-per-model selection (Option A / DEC-055), so mixed-suite comparisons are
  impossible.
- Distinguish **empty store** from **suite-version mismatch** as operationally
  different outcomes.
- Keep results **immutable and append-only**; exclusion never deletes history.
- Record **DEC-054** and **DEC-055** in `docs/DECISIONS.md`.
- Freeze Phase A Compatibility Invariants that OS-5 must not violate (score
  ordinal, viable-only viability, advisor-owned weights, alias lifecycle).

**Not BREAKING.** No historical JSON rewrite. No regeneration of
`benchmarks/prompts/standard.json`. No advisor scoring or VRAM rename (OS-6).

## Capabilities

### New Capabilities

- _(none — suite identity and comparability extend the existing `platform`
  capability)_

### Modified Capabilities

- `platform`: Add verified suite identity (pinned canonicalization, shared
  canonicalizer, fail-loud load), suite-aware storage selection (Option A),
  empty-vs-mismatch distinction, append-only archival invariants, and Phase A
  Compatibility Invariants that constrain this change and the OS-5/OS-6 boundary.

## Impact

| Area | Impact |
|---|---|
| `src/inference_x/benchmarks/suite_identity.py` | **New (OS-5-owned).** Sole canonicalizer. |
| `src/inference_x/benchmarks/runner.py` | **Shared with OS-6.** OS-5 owns `_load_suite` verification only. |
| `src/inference_x/benchmarks/storage.py` | **OS-5-owned.** Suite filter before latest-per-model; empty vs mismatch. |
| `scripts/suite_version.py` + `Makefile` | **OS-5-owned.** Regeneration entry point → shared canonicalizer. |
| `benchmarks/prompts/standard.json` | **Read-only.** Digest already correct; do not regenerate. |
| `benchmarks/results/*.json` | **Read-only.** Immutable / append-only; no rewrite. |
| `src/inference_x/benchmarks/advisor.py` | **Out of bounds.** OS-6 owns scoring; OS-5 must not touch. |
| `docs/DECISIONS.md` | **Shared (append-only).** DEC-054, DEC-055 only. Not DEC-056/057. |
| `tests/unit/test_benchmark_runner.py`, `tests/unit/test_storage.py` | OS-5 tests; storage tests are a shared conflict point with OS-6. |
| Chat / streaming / admission API | **None.** |

### ADRs

- **DEC-054** — Benchmark suite identity is a pinned content hash *(this change)*.
- **DEC-055** — `suite_version` is a necessary but not sufficient comparability key
  *(this change)*.
- **Do not author DEC-056 or DEC-057** (OS-6).

### Sequencing

Land OS-5 before OS-6. Shared-file conflict: `runner.py`, storage tests,
`DECISIONS.md`. Filter stays in `storage.py` so `advisor.py` remains free for OS-6.

## Non-Goals

Explicitly forbidden in this change:

- Advisor scoring changes, weight recalibration, or `quant_score` deletion
- VRAM metric renaming or alias introduction/removal
- Benchmark migration or rewriting historical result JSON
- Regenerating or editing `benchmarks/prompts/standard.json` contents
- Unicode NFC/NFD normalization
- Runtime / engine version in the suite hash
- Manifest / Phase B / Phase C work
- Result pruning or archival tooling
- Putting suite filtering in `advisor.py`
