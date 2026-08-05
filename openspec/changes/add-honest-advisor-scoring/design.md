# Design — OS-6 honest advisor scoring

Implementation mechanics for the **frozen** OS-6 architecture. No redesign. Where
DEC-056 or DEC-057 owns a truth, this document names it and stops.

## Context

Today the advisor scores:

```
(0.40·throughput + 0.30·ttft + 0.20·vram + 0.10·quant_score) × 100
```

with `quant_score = 1.0` always. That constant is an accidental floor separating
viable-worst (`≥ 10`) from non-viable (`0`). The VRAM field
`peak_vram_delta_gb` is written from `_peak_vram_footprint_gb` =
`total − min(free)` — whole-device occupancy, not a model footprint.

Normative freeze: exact weights `4/9`, `1/3`, `2/9`; canonical field
`vram_device_occupied_gib`; aliases through Phase A; HardwareProfile unchanged;
no normalization/TTFT/calibration/component-score work.

## Goals / Non-Goals

**Goals:**

1. Score composed only of measured quantities under exact frozen weights.
2. `viable` sole viability signal; score is a within-report ordinal used only to
   rank viable models produced from the same benchmark suite. Score is NOT
   portable across benchmark suites, NOT portable across hardware, NOT portable
   across future weighting changes, and NOT an absolute quality metric.
3. Canonical VRAM name matches device-occupancy measurement; aliases load history.
4. Record DEC-056 and DEC-057.

**Non-Goals:**

- Suite identity / storage filtering (OS-5)
- Rewriting historical benchmark JSON
- HardwareProfile rename
- Normalization redesign, TTFT redesign, calibration, component-score API
- Phase B / Phase C
- Fixing `_check_vram_budget` occupancy-vs-estimate mismatch (record only)

## Ownership

| Concern | Owner | Not the owner |
|---|---|---|
| Scoring weights and formula | `advisor.py` | runner, storage, schemas, suite_identity |
| `viable` / `score` semantics | `advisor.py` | clients must not invent proxies |
| Canonical VRAM field + aliases | `schemas.py` | HardwareProfile |
| Construct measured occupancy | `runner.py` (write path) | advisor |
| Suite identity / suite filter | OS-5 (`suite_identity`, `storage`) | this change |

**Rule (C-weights):** The advisor owns the scoring weights. The runner must never
embed them. Storage must never embed them. Schemas must never embed them.

## Decisions

### D1 — Delete `quant_score`; exact weights (DEC-056)

```
score = (
    (4/9) * throughput_score
  + (1/3) * ttft_score
  + (2/9) * vram_score
) * 100.0   # display scale retained; score is a within-report ordinal used only to rank viable models produced from the same benchmark suite
```

Component scores remain the existing relative normalizations (unchanged scheme):

- `throughput_score = throughput / max_throughput`
- `ttft_score = 1 − ttft / max_ttft`
- `vram_score` = remaining headroom fraction (CPU-only → `1.0`)

Weights are an **uncalibrated editorial preference** with no external referent —
declared, not inferred. No calibration procedure.

**Affine-invariance consequence:** old score `100·S + 10`, new `(100/0.9)·S` (with
`S` the weighted sum of the three measured components under the old 0.4/0.3/0.2
mix before the constant). Under the frozen exact fractions, ranking among
**viable** models is unchanged because the transform is strictly increasing in the
measured weighted sum. Pin with a test.

**Floor collapse:** viable-but-worst and non-viable may both show `score == 0.0`.
They remain distinguishable **only** via `viable`. Ties at `0.0` sort by input
order (arbitrary).

### D2 — Compatibility Invariants (frozen)

| ID | Invariant |
|---|---|
| C-score | score is a within-report ordinal used only to rank viable models produced from the same benchmark suite. Score is NOT portable across benchmark suites, NOT portable across hardware, NOT portable across future weighting changes, and NOT an absolute quality metric. |
| C-viable | `viable` is the only viability signal; `score > 0` must never be used as a proxy. Every consumer uses `viable`; no consumer derives viability from score. |
| C-weights | The advisor owns the scoring weights (`4/9`, `1/3`, `2/9`). The runner must never embed them. Storage must never embed them. Schemas must never embed them. |
| C-alias | Deprecated aliases remain through Phase A; removal requires a future ADR. Canonical fields always take precedence over deprecated aliases during deserialization; if both are present with conflicting values, the canonical field wins (permanent unless superseded by a future ADR). |

Concrete ordinal pins (tests, not redesign):

- Single-result report: relative TTFT component is `0` (max = self).
- Adding a model to a report can change other models' scores.
- Score monotonicity: vary one component at a time among viable models.

### D3 — Canonical VRAM field (DEC-057)

| Role | Name |
|---|---|
| Canonical (persisted + advisor) | `vram_device_occupied_gib` |
| Deprecated alias (BenchmarkResult JSON) | `peak_vram_delta_gb` |
| Deprecated alias (AdvisorResult wire) | `vram_gb` |

**Measurement unchanged:** `total − min(free)` at sampled free-memory boundaries —
whole-device occupancy including other GPU users. Unit is already GiB; `_gb → _gib`
corrects the label only (numerically safe).

**HardwareProfile** `vram_total_gb` / `vram_free_gb` stay as-is (frozen scope
control).

**Canonical precedence (permanent):** Canonical fields always take precedence over
deprecated aliases during deserialization. If both are present with conflicting
values, the canonical field wins. This precedence rule is permanent unless
superseded by a future ADR.

**Known coupling (record, do not fix):** `_check_vram_budget` still compares this
occupancy to a per-model analytic estimate and can over-report on machines with a
display.

### D4 — Alias mechanics

- On load: accept alias-only, canonical-only, or both; on conflict, canonical wins.
- On dump (new writes): emit the canonical field `vram_device_occupied_gib`.
- Advisor response: canonical `vram_device_occupied_gib`; `vram_gb` remains as
  deprecated alias through Phase A (still serialized).
- Historical `benchmarks/results/*.json` are the real alias corpus — load all of them
  after the schema change; do not rewrite files.
- Historical JSON remains readable without migration; new JSON uses the canonical
  field; deprecated aliases remain readable throughout Phase A.

### D5 — Out-of-bounds files

Do not modify:

- `suite_identity.py`, suite verification, `make suite-version`
- `storage.py` suite-aware selection / empty-vs-mismatch
- Prompt suite contents
- Any Phase B/C surfaces

## Repository impact

| File | Ownership | Reason | Conflict risk |
|---|---|---|---|
| `benchmarks/advisor.py` | OS-6 | Weights, score, viable, VRAM read | None if OS-5 left it alone |
| `benchmarks/schemas.py` | OS-6 | Canonical field + aliases | Low |
| `benchmarks/runner.py` | OS-5 / OS-6 | OS-6: construct canonical VRAM field | **High — rebase on OS-5** |
| `tests/unit/test_advisor.py` | OS-6 | Floor, ordinal, ranking, aliases | Low |
| `tests/unit/test_benchmark_runner.py` | Shared | Write-site field name | Medium |
| `tests/unit/test_storage.py` | OS-5 / OS-6 | Alias corpus vs suite tests | **Medium — rebase on OS-5** |
| `docs/DECISIONS.md` | Both | DEC-056, DEC-057 append-only | Medium |
| `benchmarks/results/*.json` | read-only | Alias corpus | None |
| `storage.py` / `suite_identity.py` | OS-5 | Out of bounds | N/A — do not touch |

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Consumers used `score > 0` as viability | DEC-056 + test: viable-worst and gated both at `0.0`, distinguishable by `viable` |
| `vram_footprint_gib`-style misnaming | Frozen name `vram_device_occupied_gib`; DEC-057 records referent |
| Alias missed → 25 results unloadable | Load-all-25 acceptance check via alias |
| Dual values diverge | Canonical wins permanently (unless future ADR) |
| Shared `runner.py` merge with OS-5 | Land OS-5 first; OS-6 touches construction only |
| Score still looks absolute (0–100) | C-score exact wording + ordinal/monotonicity tests |

## Migration Plan

**There is no migration of stored files.**

1. Historical benchmark JSON remains readable without migration (alias).
2. Historical benchmark JSON is never rewritten.
3. New benchmark JSON uses the canonical field `vram_device_occupied_gib`.
4. Deprecated aliases remain readable throughout Phase A.
5. Rollback = revert code only.

## Open Questions

None. Architecture is frozen. DEC-054/055 are OS-5; do not reopen.
