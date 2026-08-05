# Tasks — OS-5 Verify benchmark suite identity

Dependency order. Each task carries **Why**, **Implementation**, and **Verification**.
Ownership is marked **OS-5-owned** or **Shared**. Shared-file conflict points are
called out explicitly.

## Legend

| Tag | Meaning |
|---|---|
| **OS-5-owned** | This change is the sole writer for the duration of OS-5. |
| **Shared** | Also touched by OS-6 (or other units). Land OS-5 first; rebase OS-6. |
| **Conflict** | Known contention — coordinate sequencing. |

---

## 0. Preconditions

- [x] 0.1 Confirm OS-1–OS-4 are archived / landed and no competing OS-5 change exists.
      **Why:** Suite identity must not race another Phase A unit on the same files.
      **Implementation:** Inspect `openspec/changes/` and `openspec/changes/archive/`.
      **Verification:** This change directory is the only active OS-5 OpenSpec.

- [x] 0.2 Confirm DEC-054 and DEC-055 are unused numbers in `docs/DECISIONS.md`.
      **Why:** This change authors those ADRs; collisions are silent and expensive.
      **Implementation:** Grep `### DEC-05` in `docs/DECISIONS.md`.
      **Verification:** Highest accepted DEC is DEC-053; 054/055 absent.

- [x] 0.3 Record unit-test baseline.
      **Why:** Regressions must be attributable to this change.
      **Implementation:** `uv run pytest tests/unit -q` and note pass/xfail counts.
      **Verification:** Baseline recorded in the PR description.

---

## 1. Decision records — `docs/DECISIONS.md` (**Shared**, append-only)
**Conflict:** OS-6 will append DEC-056/057 later. Do not author those here.

- [x] 1.1 Add **DEC-054**, status `accepted`: benchmark suite identity is a pinned
      content hash.
      **Why:** AGENTS.md requires an accepted ADR for a contract this durable.
      **Implementation:** Record the pinned `json.dumps` form; hash `data["prompts"]`
      only; bare hex; characterization digest `b47066…`; rejected alternatives
      (semver, git blob, signed manifest); NFC limitation; fail-loud vs DEC-047 §4;
      supersession of the plan's migration premise (no regeneration).
      **Verification:** Entry exists; cites the exact canonicalization; states no
      migration.

- [x] 1.2 Add **DEC-055**, status `accepted`: `suite_version` is necessary but not
      sufficient for comparability.
      **Why:** Prevents treating suite hash as full provenance.
      **Implementation:** Record Option A (filter in storage before latest-per-model);
      immutable/append-only archival; mixed suites excluded never deleted; input
      identity ≠ runtime identity.
      **Verification:** Entry exists; states Option A and necessary-but-not-sufficient.

- [x] 1.3 Do not author DEC-056 or DEC-057; do not reopen DEC-047–053.
      **Why:** OS-6 owns scoring/VRAM ADRs; earlier ADRs stay accepted.
      **Implementation:** Additive append only for 054/055.
      **Verification:** `git diff docs/DECISIONS.md` is append-only; no 056/057.

---

## 2. Shared canonicalizer — `src/inference_x/benchmarks/suite_identity.py`
**OS-5-owned**

- [x] 2.1 Create the sole canonicalizer module exporting the pinned digest function
      over an in-memory prompt collection.
      **Why:** One implementation owns suite identity (frozen ownership).
      **Implementation:** Implement
      `sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()`.
      Accept the parsed `prompts` list (not file bytes). Document exclusions
      (encoding/whitespace/key order/formatting) in the docstring.
      **Verification:** Unit test: shipped suite prompts →
      `b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4`.

- [x] 2.2 Add helper(s) to verify a suite dict / path against its stored
      `suite_version`, raising an actionable error naming `make suite-version` on
      mismatch or missing key.
      **Why:** Load path and CLI share one failure contract.
      **Implementation:** Missing key and digest mismatch both raise the same class of
      actionable error (not bare `KeyError`).
      **Verification:** Tests for match, mismatch, and missing key.

---

## 3. Suite load verification — `src/inference_x/benchmarks/runner.py`
**Shared with OS-6** — **Conflict: High**. OS-5 may touch `_load_suite` only.

- [x] 3.1 Change `_load_suite` to parse JSON, then verify via `suite_identity`.
      **Why:** Verification must happen before any run proceeds.
      **Implementation:** After `json.loads`, call the shared verifier on
      `data["prompts"]` vs `data.get("suite_version")`; return verified
      `(suite_version, prompts)`. Do not inline hashing.
      **Verification:** Grep shows no `hashlib` / canonical `json.dumps` in
      `runner.py`; mismatch test raises and names `make suite-version`.

- [x] 3.2 Do not modify prompt measurement, VRAM fields, or result construction
      beyond suite load.
      **Why:** Those lines are OS-2/OS-6 contention surfaces.
      **Implementation:** Diff confined to suite-load / import lines.
      **Verification:** `git diff` on `runner.py` excludes VRAM / scoring / token paths.

---

## 4. Regeneration entry point — `scripts/suite_version.py` + `Makefile`
**OS-5-owned**

- [x] 4.1 Add `scripts/suite_version.py` that loads a suite JSON, hashes prompts via
      `suite_identity`, and prints / optionally writes the digest.
      **Why:** Operators need one commanded regeneration path.
      **Implementation:** Import shared canonicalizer only; no duplicate algorithm.
      **Verification:** Script output for `benchmarks/prompts/standard.json` equals
      `b47066…`.

- [x] 4.2 Add `make suite-version` invoking that script.
      **Why:** Error messages name this target.
      **Implementation:** Makefile target delegates to the script.
      **Verification:** `make suite-version` succeeds and prints the characterization
      digest for the shipped suite.

- [x] 4.3 Do not edit `benchmarks/prompts/standard.json`.
      **Why:** Digest already correct; this is verification, not migration.
      **Implementation:** No write to the suite file in default verify flow.
      **Verification:** `git diff benchmarks/prompts/standard.json` is empty.

---

## 5. Storage filtering — `src/inference_x/benchmarks/storage.py`
**OS-5-owned** (tests shared with OS-6)

- [x] 5.1 Filter by expected `suite_version` before latest-per-model selection.
      **Why:** Mixed-suite comparisons must be impossible (Option A / DEC-055).
      **Implementation:** Accept expected suite version; filter `all_results()` first;
      then compute latest per model on the filtered set. Do not put this in
      `advisor.py`.
      **Verification:** Fixture with two suite versions for one model returns only the
      expected suite's latest.

- [x] 5.2 Distinguish empty store from suite-version mismatch.
      **Why:** Operationally different; silent empty is a lie.
      **Implementation:** Selection API exposes a distinct status/signal for
      `empty` vs `suite_mismatch` vs `ok` (exact shape per design D4).
      **Verification:** Tests assert the two failure outcomes are not identical.

- [x] 5.3 Never delete or rewrite result files when filtering.
      **Why:** Append-only archival invariant.
      **Implementation:** Filter in memory only.
      **Verification:** After mismatch selection, non-matching files still exist on
      disk unchanged.

- [x] 5.4 Wire thin advise consumers to pass expected suite version and surface
      mismatch status without reimplementing the filter.
      **Why:** Distinction must reach operators; advisor ranking stays untouched.
      **Implementation:** Minimal edits to advise entry points only (route/script).
      Expected suite comes from the shared canonicalizer / verified suite load.
      **Verification:** Advise path with only mismatched results does not look identical
      to a truly empty store; `advisor.py` diff is empty.

---

## 6. Compatibility guardrails (**OS-5 must not violate**)

- [x] 6.1 Confirm `advisor.py` is untouched.
      **Why:** Advisor owns weights/score/viable (C-weights, C-score, C-viable); OS-6
      file.
      **Implementation:** No edits.
      **Verification:** `git diff -- src/inference_x/benchmarks/advisor.py` empty.

- [x] 6.2 Confirm benchmark collection does not embed advisor weights.
      **Why:** C-weights — collection never depends on advisor weights.
      **Implementation:** Runner / suite_identity / storage import no scoring weights.
      **Verification:** Grep OS-5-touched modules for weight literals / advisor score
      imports — none related to scoring.

- [x] 6.3 Confirm no VRAM rename and no alias removal.
      **Why:** C-alias; VRAM rename is OS-6 / DEC-057; alias removal needs future ADR.
      **Implementation:** No schema field renames; no alias deletions.
      **Verification:** `peak_vram_delta_gb` (and any existing aliases) still present;
      no `vram_device_occupied_gib` introduction in this change.

- [x] 6.4 Confirm historical result JSON is read-only.
      **Why:** No migration; rollback never rewrites history.
      **Implementation:** Do not rewrite `benchmarks/results/*.json`.
      **Verification:** `git diff -- benchmarks/results` empty.

---

## 7. Tests

- [x] 7.1 Canonicalization tests (**OS-5-owned**, `tests/unit/test_benchmark_runner.py`
      and/or new `test_suite_identity.py`).
      **Why:** Pin algorithm and characterization proof.
      **Implementation:** Digest equality; formatting independence; content/order
      sensitivity; `suite_version` key excluded from hash.
      **Verification:** All scenarios in the platform spec for pinned hash pass.

- [x] 7.2 Mismatch / missing-key load tests.
      **Why:** Fail-loud contract.
      **Implementation:** Temporary suite files covering match, mismatch, missing key.
      **Verification:** Raises; message contains `make suite-version`.

- [x] 7.3 Storage filtering tests (**Shared** with OS-6 — **Conflict: Medium**).
      **Why:** Option A + empty vs mismatch.
      **Implementation:** Fixtures for mixed suites, matching latest-per-model, empty
      dir, mismatch-only dir.
      **Verification:** Spec scenarios for storage filter and distinct outcomes pass.

- [x] 7.4 Shared-implementation test.
      **Why:** Prevent duplicate hash drift.
      **Implementation:** Assert regeneration path and verifier use the same function
      object / module API.
      **Verification:** One canonicalizer symbol; no second algorithm in scripts.

- [x] 7.5 Regression: existing unit suite still green; existing stored results still
      load under the current suite digest.
      **Why:** No-migration claim must hold for the 25 on-disk results.
      **Implementation:** Load `benchmarks/results` through suite-aware selection with
      the shipped digest.
      **Verification:** Selection status `ok` (or equivalent); models still appear;
      file contents unchanged.

---

## 8. Validation (exhaustive — run before marking complete)

### 8.1 Canonicalization verification
- [x] V1 Computed digest of shipped prompts equals
      `b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4`.
- [x] V2 Reordered object keys / altered whitespace in a round-tripped suite file
      still verifies when prompts are unchanged.
- [x] V3 Changing one prompt `text` or `label`, or prompt order, changes the digest.

### 8.2 Digest verification at load
- [x] V4 Matching suite loads and returns `(suite_version, prompts)`.
- [x] V5 Mismatched `suite_version` raises and names `make suite-version`.
- [x] V6 Missing `suite_version` raises actionable error (not bare `KeyError`).

### 8.3 Shared canonicalizer
- [x] V7 `make suite-version` output matches verifier digest for the same file.
- [x] V8 No second hash implementation in `runner.py` or `scripts/suite_version.py`.

### 8.4 Storage filtering
- [x] V9 Mixed-suite corpus → selection returns only expected suite.
- [x] V10 Latest-per-model among matching suite versions still prefers newest
      timestamp.
- [x] V11 Advisor is never handed mixed suites from the selection path
      (`advisor.py` unchanged).

### 8.5 Empty vs mismatch
- [x] V12 Empty results directory → empty-store outcome.
- [x] V13 Only mismatched suite versions on disk → suite-mismatch outcome,
      distinguishable from V12.

### 8.6 Rollback / immutability
- [x] V14 Non-matching result files remain on disk after selection.
- [x] V15 `git diff -- benchmarks/prompts/standard.json benchmarks/results` empty.
- [x] V16 Documented rollback = revert commits; no historical JSON rewrite steps.

### 8.7 Compatibility
- [x] V17 `advisor.py` diff empty.
- [x] V18 No advisor weight literals in runner / suite_identity / storage.
- [x] V19 No VRAM field rename; no alias removal.
- [x] V20 Score / viable semantics not redefined by OS-5.

### 8.8 Regression
- [x] V21 `uv run pytest tests/unit -q` matches or improves on baseline (no new
      unexpected xfail).
- [x] V22 Existing stored results for the current suite still select successfully.
- [x] V23 Chat completion / health / streaming unit tests unaffected.

### 8.9 ADR / OpenSpec hygiene
- [x] V24 DEC-054 and DEC-055 present and `accepted`.
- [x] V25 DEC-056 and DEC-057 absent from this change.
- [x] V26 `openspec validate verify-benchmark-suite-identity --strict` passes.

---

## 9. Out of scope checklist (must remain undone)

- [ ] Confirm no advisor scoring / weight recalibration tasks were added.
- [ ] Confirm no VRAM metric renaming tasks were added.
- [ ] Confirm no benchmark migration / historical JSON rewrite tasks were added.
- [ ] Confirm no Phase B or Phase C tasks were added.
- [ ] Confirm suite filtering was not placed in `advisor.py`.
