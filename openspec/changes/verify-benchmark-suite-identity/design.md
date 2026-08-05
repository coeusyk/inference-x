# Design — OS-5 verify benchmark suite identity

Implementation mechanics for the **frozen** OS-5 architecture. No redesign. Where
an ADR owns a truth, this document names it and stops.

## Context

Today `_load_suite` reads `data["suite_version"]` verbatim. Nothing hashes the
prompts. `latest_per_model()` selects by timestamp alone, so results from different
suites can reach the advisor. The shipped suite literal already equals the pinned
digest:

```
b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4
```

OS-5 restores verification and adds the consumption-side filter. It does not migrate
history.

Normative inputs: frozen architecture (DEC-054 / DEC-055 direction),
`docs/REVIEW-2026-08-04-os5-os6-finalization.md` §0 (premise correction), and the
compatibility freeze for Phase A invariants that OS-5 must not violate.

## Goals / Non-Goals

**Goals:**

1. One shared canonicalizer owns suite identity.
2. Load-time verification fails loudly on missing or mismatched `suite_version`.
3. Storage Option A: filter by expected suite before latest-per-model.
4. Empty store ≠ suite mismatch.
5. Results remain immutable and append-only.
6. Record DEC-054 and DEC-055; preserve Compatibility Invariants.

**Non-Goals:**

- Advisor scoring / weights / `quant_score` / VRAM rename (OS-6; DEC-056/057)
- Algorithm-tag migration of stored digests (bare hex retained)
- Unicode normalization
- Benchmark migration or rewriting historical JSON
- Regenerating `standard.json`
- Phase B / Phase C
- Filter logic in `advisor.py`

## Ownership

| Concern | Owner | Not the owner |
|---|---|---|
| Canonical hash of prompts | `suite_identity` module | runner inline, scripts inline, storage |
| Verify at suite load | `runner._load_suite` via shared module | advisor |
| Regenerate digest CLI | `scripts/suite_version.py` / `make suite-version` | ad-hoc shell one-liners |
| Suite-aware selection | `storage.ResultStore` | `advisor.py` |
| Empty vs mismatch surface | storage API (+ thin advise consumers) | advisor ranking logic |
| Scoring weights / score / viable | advisor (OS-6; unchanged here) | runner, storage, suite_identity |
| VRAM field names / aliases | schemas / OS-6 | this change |

**Rule:** One implementation hashes. Storage selects. Advisor ranks what it is handed.
OS-5 must not teach the runner about advisor weights.

## Decisions

### D1 — Hashed object (DEC-054)

Canonicalization operates on the **in-memory prompt collection after JSON parsing**.

Pinned form:

```python
sha256(
    json.dumps(
        prompts,  # list[{"label", "text"}, ...] — data["prompts"] only
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
```

**In identity:** prompt ordering and prompt values (`label`, `text`).

**Excluded from identity:** file encoding, whitespace, indentation, line endings,
JSON object key order, and serialization formatting. Switching suite file format
(e.g. YAML) cannot redefine identity without changing the parsed prompt collection
and a new ADR.

`suite_version` is excluded from the hashed content — it lives in the file it
identifies. Hash `data["prompts"]`, never the whole file.

Stored form remains **bare 64-char hex**. No `sha256:` prefix migration.

Limitation (recorded, not fixed): byte-identity of canonical JSON; NFC vs NFD
spellings of the same characters hash differently. Suite is ASCII.

### D2 — Single shared canonicalizer

Module: `src/inference_x/benchmarks/suite_identity.py` (sole implementation).

All producers and consumers **must** call it:

- `runner._load_suite` verification
- `scripts/suite_version.py` / `make suite-version`
- any future consumer

No duplicated hash implementations. No inline `hashlib` / `json.dumps` copy in
runner or scripts.

### D3 — Fail-loud load posture

Mismatch or missing `suite_version` raises with an actionable message naming
`make suite-version`. Not `KeyError`. Not silent accept.

This does **not** contradict DEC-047 §4: fail-open applies to the serving admission
path with a live client; the benchmark harness is first-party tooling where a silent
wrong number is worse than a stopped run. DEC-054 records the contrast.

### D4 — Storage Option A (DEC-055)

Filter by expected `suite_version` **before** latest-per-model selection.

```
all_results()
    → keep only result.suite_version == expected
    → latest_per_model on the filtered set
```

Mixed-suite rankings never reach the advisor. `advisor.py` stays out of bounds.

Widen the storage selection API only as needed to:

1. Accept the expected suite version.
2. Return selected results.
3. Expose whether the outcome is **empty store**, **ok**, or **suite mismatch**
   (results exist on disk, but none match the requested suite).

Thin consumers (`/v1/benchmark/advise`, `scripts/advise.py`) may surface the
mismatch status; they must not reimplement filtering.

### D5 — Append-only archival

Results remain immutable and append-only. Incomparable results are **excluded by
the consumer**, never deleted. Rollback never requires rewriting historical
benchmark files — revert code; disk history is untouched.

### D6 — Necessary but not sufficient

`suite_version` pins the benchmark **input**, not the runtime that consumed it.
Hardware matching (DEC-036) and other dimensions remain separate. OS-5 does not
claim full provenance.

### D7 — Characterization proof

Acceptance proof of no-migration: computing the pinned digest over the shipped
suite's prompts equals

`b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4`

If this fails, stop — the canonicalization is wrong.

### D8 — Compatibility Invariants (Phase A freeze)

OS-5 SHALL NOT violate these. They are recorded here and in the spec so OS-5
cannot accidentally redefine OS-6's surface:

| ID | Invariant |
|---|---|
| C-score | `score` is a within-report ordinal only — not absolute, not portable across reports, not a 0–100 quality scale. |
| C-viable | `viable` is the only viability signal; `score > 0` must not be used as a proxy. |
| C-weights | The advisor owns scoring weights (`4/9`, `1/3`, `2/9` under the frozen OS-6 plan). Benchmark collection never embeds or assumes them. |
| C-alias | Deprecated aliases remain through Phase A; removal requires a future ADR. |

OS-5 does not implement OS-6 scoring or VRAM rename. It must leave advisor weights,
score semantics, viable semantics, and VRAM field names untouched.

## Repository impact

| File | Ownership | Reason | Conflict risk |
|---|---|---|---|
| `src/inference_x/benchmarks/suite_identity.py` | OS-5 | Sole canonicalizer | None |
| `src/inference_x/benchmarks/runner.py` | OS-5 / OS-6 | OS-5: verify on load. OS-6: VRAM construction | **High — land OS-5 first** |
| `src/inference_x/benchmarks/storage.py` | OS-5 | Suite-aware selection; empty vs mismatch | Medium — shared tests |
| `src/inference_x/benchmarks/advisor.py` | OS-6 | Out of bounds for OS-5 | None if filter stays in storage |
| `scripts/suite_version.py` | OS-5 | Calls shared canonicalizer only | None |
| `Makefile` | OS-5 | `suite-version` target | Low |
| `benchmarks/prompts/standard.json` | read-only | Digest already correct | None |
| `benchmarks/results/*.json` | read-only | Append-only history | None |
| `tests/unit/test_benchmark_runner.py` | OS-5 | Canonicalization + mismatch | Low |
| `tests/unit/test_storage.py` | OS-5 / OS-6 | Suite mismatch + later alias corpus | **Medium — rebase OS-6** |
| `docs/DECISIONS.md` | Both | DEC-054, DEC-055 only here | Medium — append-only |
| Advise route / `scripts/advise.py` | thin consumer | Surface mismatch status only | Low |

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Implementer picks a different canonicalization and breaks history | DEC-054 pins the form; equality test vs `b47066…` |
| Verification with no consumer | Option A filter in storage (D4) |
| Filter lands in `advisor.py`, blocking OS-6 parallelism | Ownership table; advisor out of bounds |
| Fail-loud "corrected" to fail-open via DEC-047 §4 | DEC-054 records harness vs serving posture |
| Duplicate hash in script vs runner | Single module; tests assert shared call path |
| Empty and mismatch look identical to operators | Distinct selection status (D4) |

## Migration Plan

**There is no migration.**

1. Land code + tests; do not edit `standard.json` or result files.
2. Existing 25 results share the current suite digest → filter is a no-op on them.
3. **Rollback:** revert the change. Historical files need no rewrite.

## Open Questions

None. Architecture is frozen. DEC-056 / DEC-057 are OS-6 and out of scope.
