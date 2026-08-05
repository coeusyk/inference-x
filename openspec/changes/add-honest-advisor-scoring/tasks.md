# Tasks — OS-6 Honest advisor scoring and device-occupancy VRAM

Dependency order. Each task carries **Why**, **Implementation**, and **Verification**.
Ownership is marked **OS-6-owned** or **Shared**. Shared-file conflicts with OS-5 are
called out explicitly.

## Legend

| Tag | Meaning |
|---|---|
| **OS-6-owned** | This change is the sole writer for the duration of OS-6. |
| **Shared** | Also touched by OS-5. Rebase OS-6 on landed OS-5. |
| **Conflict** | Known contention with OS-5. |
| **Out of bounds** | Must not be edited. |

---

## 0. Preconditions

- [x] 0.1 Confirm OS-5 (`verify-benchmark-suite-identity`) is merged / archived and
      DEC-054 / DEC-055 are `accepted`.
      **Why:** Shared `runner.py` / storage tests / `DECISIONS.md` must not race.
      **Implementation:** Inspect git history and `docs/DECISIONS.md`.
      **Verification:** OS-5 suite identity and suite-aware selection already present;
      this change does not reopen them.

- [x] 0.2 Confirm DEC-056 and DEC-057 are unused numbers.
      **Why:** This change authors those ADRs only.
      **Implementation:** Grep `### DEC-05` in `docs/DECISIONS.md`.
      **Verification:** 056/057 absent before this change.

- [x] 0.3 Record unit-test baseline.
      **Why:** Attribute regressions.
      **Implementation:** `uv run pytest tests/unit -q`.
      **Verification:** Pass/xfail counts recorded in the PR.

---

## 1. Decision records — `docs/DECISIONS.md` (**Shared**, append-only)
**Conflict with OS-5:** append after DEC-054/055. Do not edit OS-5 ADRs.
Do **not** author any ADR beyond DEC-056 and DEC-057.

- [x] 1.1 Add **DEC-056**, status `accepted`: advisor score reflects measured
      quantities only.
      **Why:** Durable contract for weights, floor collapse, ordinal score, viable-only.
      **Implementation:** Record deletion of `quant_score`; exact weights `4/9`,
      `1/3`, `2/9`; affine-invariance / ranking preservation among viable models;
      floor collapse (`viable` sole signal); ties at `0.0`; weights uncalibrated;
      score is a within-report ordinal used only to rank viable models produced from
      the same benchmark suite; score is NOT portable across benchmark suites, NOT
      portable across hardware, NOT portable across future weighting changes, and NOT
      an absolute quality metric (single-result TTFT component zero; adding a model
      can change others).
      **Verification:** Entry exists and states all of the above.

- [x] 1.2 Add **DEC-057**, status `accepted`: VRAM number is device occupancy.
      **Why:** Name must match measurement referent.
      **Implementation:** Record `vram_device_occupied_gib`; measurement
      `total − min(free)`; aliases `peak_vram_delta_gb` / advisor `vram_gb` through
      Phase A; HardwareProfile `*_gb` unchanged; unit label correction (already GiB);
      known `_check_vram_budget` coupling recorded not fixed; canonical fields always
      take precedence over deprecated aliases during deserialization — if both are
      present with conflicting values, the canonical field wins (permanent unless
      superseded by a future ADR).
      **Verification:** Entry exists; no HardwareProfile rename decision deferred as
      "out of scope — frozen unchanged".

- [x] 1.3 Do not author any additional ADRs; do not reopen DEC-047–055.
      **Why:** Freeze forbids extra ADRs.
      **Implementation:** Additive append of 056/057 only.
      **Verification:** Diff is append-only for those two entries.

---

## 2. Schemas — `src/inference_x/benchmarks/schemas.py` (**OS-6-owned**)

- [x] 2.1 Rename `BenchmarkResult.peak_vram_delta_gb` → canonical
      `vram_device_occupied_gib` with deprecated alias `peak_vram_delta_gb`.
      **Why:** DEC-057; historical JSON must still load.
      **Implementation:** Pydantic field + alias / validation alias; canonical wins on
      conflict (permanent precedence).
      **Verification:** Alias-only loads; canonical-only loads; equal dual values load;
      conflicting dual values resolve to the canonical value; new dumps expose
      canonical key.

- [x] 2.2 On `AdvisorResult`, add canonical `vram_device_occupied_gib` and keep
      `vram_gb` as deprecated alias through Phase A.
      **Why:** C-alias; wire compatibility.
      **Implementation:** Same alias pattern; canonical wins on conflict.
      **Verification:** Serialization exposes canonical and still serializes `vram_gb`.

- [x] 2.3 Do not rename `HardwareProfile.vram_total_gb` / `vram_free_gb`.
      **Why:** Frozen decision 8.
      **Implementation:** No edits to those fields.
      **Verification:** Schema field names unchanged.

---

## 3. Runner write path — `src/inference_x/benchmarks/runner.py` (**Shared**)
**Conflict with OS-5: High.** Touch only VRAM construction / result field assignment.
Do not touch `_load_suite` / suite identity.

- [x] 3.1 Write `vram_device_occupied_gib` at `BenchmarkResult` construction using the
      existing device-occupancy computation.
      **Why:** New runs emit the canonical name.
      **Implementation:** Keep `total − min(free)` semantics; rename local helpers /
      kwargs only as needed for the field; do not change the numeric formula.
      **Verification:** Diff excludes suite-load code; numeric value unchanged vs prior
      helper for identical hardware snapshots.

- [x] 3.2 Leave `_check_vram_budget` comparison logic unchanged (known coupling).
      **Why:** Explicit non-goal; DEC-057 records it.
      **Implementation:** At most rename the parameter that receives the occupancy
      number; do not alter the estimate comparison.
      **Verification:** Behavior of budget exceeded / warning strings still matches
      prior tests aside from field naming.

---

## 4. Advisor — `src/inference_x/benchmarks/advisor.py` (**OS-6-owned**)

- [x] 4.1 Delete `quant_score` and apply exact weights `4/9`, `1/3`, `2/9`.
      **Why:** DEC-056; remove accidental floor.
      **Implementation:** Replace `0.40/0.30/0.20/0.10` block; keep `* 100.0` display
      scale; keep existing relative normalization helpers.
      **Verification:** No `quant_score` symbol; weights are exact fractions (not
      rounded 0.444…).

- [x] 4.2 Read `vram_device_occupied_gib` (alias-populated) instead of
      `peak_vram_delta_gb`.
      **Why:** Canonical field.
      **Implementation:** Attribute access on the schema field.
      **Verification:** Advisor tests pass with old-key fixtures via alias.

- [x] 4.3 Populate advisor output canonical VRAM + retain `vram_gb` alias.
      **Why:** C-alias.
      **Implementation:** Set canonical; alias wiring via schema.
      **Verification:** `AdvisorResult` carries both views without conflict.

- [x] 4.4 Do not change `_warm_ttft_ms`, normalization scheme, or viability gate
      structure (beyond score composition).
      **Why:** Non-goals 9, 12, 13.
      **Implementation:** Leave those helpers/logic intact.
      **Verification:** Diff limited to score composition + VRAM field reads/writes.

- [x] 4.5 Update module docstring so weights are declared as uncalibrated editorial
      preference, not reasoned inference.
      **Why:** Review honesty requirement retained by freeze.
      **Implementation:** Replace "most visible performance signal" framing.
      **Verification:** Docstring states uncalibrated + exact fractions; score is a
      within-report ordinal used only to rank viable models produced from the same
      benchmark suite; `viable` is sole viability signal.

---

## 5. Tests

- [x] 5.1 Update `tests/unit/test_advisor.py` (**OS-6-owned**).
      **Why:** Floor collapse, ordinal pins, ranking invariance, weight exactness.
      **Implementation:**
      - Update / replace `test_scores_are_normalized_0_to_100` so it does not claim
        absolute quality — score is a within-report ordinal used only to rank viable
        models produced from the same benchmark suite.
      - Add viable-worst vs VRAM-gated both at `0.0`, distinct `viable`.
      - Add single-result TTFT-component / ordinal pin.
      - Add "adding a model changes other scores" pin.
      - Add viable ranking-order invariance across weight change.
      - Add score monotonicity: vary one component at a time (throughput, TTFT, VRAM)
        while holding the others constant among viable models.
      - Point fixtures at canonical field (alias still accepted).
      **Verification:** New and updated tests pass.

- [x] 5.2 Update write-site tests in `tests/unit/test_benchmark_runner.py` (**Shared**).
      **Why:** Construction field name.
      **Implementation:** Assert canonical field where peak alias was asserted.
      **Verification:** Runner unit tests green.

- [x] 5.3 Alias corpus via storage load (**Shared** — **Conflict with OS-5: Medium**).
      **Why:** 25 historical files are the real compatibility corpus.
      **Implementation:** Test that `ResultStore.all_results()` loads every existing
      `benchmarks/results/*.json` after the schema alias lands (or equivalent
      directory load). Do not rewrite those files. Rebase carefully onto OS-5 storage
      tests.
      **Verification:** All historical files validate; `git diff benchmarks/results`
      empty.

- [x] 5.4 Confirm out-of-bounds modules untouched.
      **Why:** Forbid suite identity / storage filtering changes.
      **Implementation:** No edits to `suite_identity.py`, `storage.py` selection
      semantics, or suite JSON.
      **Verification:** `git diff` empty for those paths (except unavoidable import
      renames — prefer zero).

---

## 6. Compatibility guardrails

- [x] 6.1 Confirm runner, storage, and schemas embed no advisor weights.
      **Why:** C-weights repository ownership rule.
      **Verification:** Grep runner / suite_identity / storage / schemas for `4/9`,
      `1/3`, `2/9` scoring — none.

- [x] 6.2 Confirm aliases not scheduled for removal in this change.
      **Why:** C-alias; removal needs future ADR.
      **Verification:** Both aliases present and tested; advisor serialization still
      includes `vram_gb`.

- [x] 6.3 Confirm no historical JSON rewrite / migration tasks executed.
      **Why:** Historical JSON remains readable without migration and is never
      rewritten; new JSON uses the canonical field.
      **Verification:** `git status --porcelain benchmarks/results` empty.

- [x] 6.4 Confirm every advisor consumer uses `viable`, never score, for viability.
      **Why:** Viability ownership.
      **Verification:** Audit `/v1/benchmark/advise` and `scripts/advise.py` (and any
      other first-party consumers); no `score > 0` / score-threshold viability.

---

## 7. Validation (exhaustive — run before marking complete)

### 7.1 Scoring
- [x] V1 `quant_score` absent from `advisor.py`.
- [x] V2 Weights are exactly `4/9`, `1/3`, `2/9` (not approximate floats alone).
- [x] V3 Viable models' relative ranking unchanged vs measured-only prior ordering
      (invariance test).
- [x] V4 Normalization helpers / `_warm_ttft_ms` unchanged in behavior.
- [x] V4a Score monotonicity: vary throughput, TTFT, or VRAM alone among viable
      models; ordering follows the varied component.

### 7.2 Compatibility Invariants
- [x] V5 C-score: score is a within-report ordinal used only to rank viable models
      produced from the same benchmark suite (exact wording present in docs/spec).
- [x] V5a C-score: score is NOT portable across suites, hardware, or future
      weighting changes, and NOT an absolute quality metric.
- [x] V6 C-score: single-result ordinal pin; adding a model can change prior scores.
- [x] V7 C-viable: viable-worst and gated both may be `0.0`, distinct via `viable`.
- [x] V8 C-viable: every consumer uses `viable`; no consumer derives viability from
      score.
- [x] V9 C-weights: runner, storage, schemas, and suite_identity do not embed
      weights.
- [x] V10 C-alias: aliases present through Phase A; removal requires future ADR.

### 7.3 VRAM metric and alias round-trip
- [x] V11 New results write `vram_device_occupied_gib`.
- [x] V12 Measurement still `total − min(free)` for identical snapshots.
- [x] V13 Alias-only historical JSON loads (`peak_vram_delta_gb`).
- [x] V13a Canonical-only round-trip succeeds.
- [x] V13b Canonical + alias equal values succeed.
- [x] V13c Canonical + alias conflicting values → canonical wins.
- [x] V14 Advisor serialization includes canonical field and still serializes
      `vram_gb`.
- [x] V15 `HardwareProfile.vram_*_gb` names unchanged.
- [x] V16 `_check_vram_budget` comparison logic unchanged.

### 7.4 Migration / immutability
- [x] V17 Historical benchmark JSON remains readable without migration and is never
      rewritten (`benchmarks/results/*.json` unmodified).
- [x] V18 `benchmarks/prompts/standard.json` unmodified.
- [x] V19 Rollback documented as code revert only; new JSON uses the canonical field.

### 7.5 OS-5 boundary
- [x] V20 `suite_identity.py` untouched.
- [x] V21 `storage.py` suite-aware selection semantics untouched.
- [x] V22 No suite identity / filtering regressions
      (`pytest tests/unit/test_suite_identity.py tests/unit/test_storage.py`).

### 7.6 Tooling
- [x] V23 `uv run pytest tests/unit -q` green (no new unexpected xfail).
- [x] V24 `uv run ruff check` on touched paths green.
- [x] V25 `uv run mypy` on touched packages green.
- [x] V26 `openspec validate add-honest-advisor-scoring --strict` passes.

### 7.7 ADR hygiene
- [x] V27 DEC-056 and DEC-057 present and `accepted`.
- [x] V28 No additional ADRs authored by this change.

---

## 8. Out of scope checklist (must remain undone)

- [ ] No Phase B / Phase C tasks.
- [ ] No suite identity changes.
- [ ] No storage filtering changes.
- [ ] No benchmark migration / historical JSON rewrite.
- [ ] No normalization redesign.
- [ ] No calibration procedure.
- [ ] No component-score API.
- [ ] No TTFT redesign.
- [ ] No HardwareProfile rename.
