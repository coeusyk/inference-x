# Phase A Finalization Review — OS-5 and OS-6

**Date:** 2026-08-04
**Scope:** Are OS-5 (benchmark suite integrity) and OS-6 (advisor honesty) still the
correct final Phase A milestones?
**Status of inputs:** OS-1 – OS-4 complete and frozen. `docs/PHASE-A-EXECUTION-PLAN.md`
is the plan under review.
**Verdict:** Both units remain correct in *concern*. Each has one under-specified
boundary that must be closed before the OpenSpec is authored. One factual premise of
the plan is wrong and makes OS-5 materially cheaper than planned.

No external research was run. The questions here are about repository-internal
heuristics and weight choices; no external corpus settles what `0.40 × throughput`
should be for this project.

---

## 0. The finding that reorganizes OS-5

The plan (§457, §256, §808, §889) assumes that computing `suite_version` will produce
a *new* value, invalidating stored results, and lists the regenerated hash as a
rollback hazard.

**That premise is false.** Verified by direct computation:

```
stored suite_version : b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4
sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False,
                  separators=(",", ":")).encode("utf-8"))
                     : b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4
                       → MATCH
```

The value in `benchmarks/prompts/standard.json` **is** the correct SHA-256 of the
prompt list under a compact-JSON canonicalization. The hash was computed once,
correctly, and then the computation was dropped from the code. What is missing is the
*verification*, not the *value*.

This does not overturn the earlier reviews. `UNDERSTANDING-INFERENCE-X.md:540`
("opaque literal, not a computed hash") and `REVIEW-2026-08-03-architecture.md:26`
("no `sha256` call anywhere in `src/` or `scripts/`") are both correct about the code
as it stands. The probe disproves only the stronger inference — that the literal is
arbitrary and that restoring the computation must break history.

Consequences:

| Plan statement | Status |
|---|---|
| §457 "regenerated value in `benchmarks/prompts/standard.json`" | **Void.** Nothing regenerates. |
| §808 rollback hazard 2, "OS-5's regenerated `standard.json` hash" | **Void.** The hazard does not exist. |
| §256 / §889 "recomputed `suite_version` rendering stored results incomparable" | **Void.** All 25 stored results stay comparable. |
| §3 item 5 "ADR noting pre-existing stored results are now correctly marked incomparable" | **Inverted.** The ADR records that they remain comparable, and why. |

The risk profile flips with it. The risk is no longer "OS-5 breaks 25 stored results."
It is **"whoever implements OS-5 picks a canonicalization that is not the one already
in the file, and manufactures a break that did not need to happen."**

### The canonicalization must be pinned, and the pin proved

Three candidate canonicalizations collide on `b47066…` today, purely by accident: every
prompt is ASCII (so `ensure_ascii` is inert) and `label` sorts before `text` (so
`sort_keys` is inert). Add one non-ASCII prompt with keys in the other order and they
diverge — verified:

```
pin (sort_keys=True,  ensure_ascii=False)  a5e4c390a9ace527…
     sort_keys=False, ensure_ascii=False   26546a3c7c1e3e2e…
     sort_keys=True,  ensure_ascii=True    e9d6b4cfb4798180…
     sort_keys=False, ensure_ascii=True    c027aa006d4350d5…
                                           → 4 distinct hashes of 4
```

That divergence is the whole argument for pinning. The recommended canonical form is
the one that is simultaneously the most robust *and* byte-identical to what is already
stored:

> `sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False,
>   separators=(",", ":")).encode("utf-8")).hexdigest()`

- `sort_keys=True` — key insertion order in the JSON file must not change identity.
- `ensure_ascii=False` + explicit UTF-8 — a non-ASCII prompt hashes by its characters,
  not by its `\uXXXX` escape spelling.
- `separators=(",", ":")` — formatting/whitespace must not change identity.
- The `suite_version` key is **excluded** from the hashed content: it lives inside the
  file it identifies. Hash `data["prompts"]`, not the file.

Honest limitation, stated and not fixed: this is byte-identity of the canonical JSON.
Unicode normalization defeats it — an NFC vs NFD spelling of the same prompt yields
different hashes (verified). Adding NFC normalization is scope creep for a suite that
is 100% ASCII. Record the limitation; do not build for it.

**The acceptance criterion writes itself, and it is also the migration strategy:** a
test asserting that the computed hash of the shipped suite equals the literal
`b47066…`. That test *proves no break occurred*. There is nothing else to migrate.

---

## 1. Architectural review

### 1.1 OS-5 — Benchmark suite integrity

**Is content hashing the right architecture?** Yes. The alternatives were considered
and each is worse:

| Alternative | Why it loses |
|---|---|
| Hand-bumped semantic version (`v1`, `v2`) | This is precisely the current failure — a self-declared string nothing verifies. Trades one honesty defect for the same one with better ergonomics. |
| Git blob hash / commit of the suite file | Couples benchmark identity to VCS state. Breaks outside a checkout (release tarball, container). Changes on whitespace-only edits, so it over-invalidates. |
| Signed manifest | There is no adversary in Phase A. Signing defends against tampering; the actual problem is accidental drift. Wrong threat model. |
| Content hash of a pinned canonical form | Deterministic, VCS-independent, offline-verifiable, invalidates exactly when semantics change. |

Content hashing is correct. Strengthen it with the pin in §0 and one format decision:
store the value **algorithm-tagged** — `"sha256:b47066…"` — rather than bare hex.
Cost today is one `split(":", 1)` in the verifier and a one-line accept-bare-hex
compatibility path for the 25 stored results. Benefit: the field can name a different
algorithm later without a second field or a guess-by-length heuristic. This is the one
place where a small forward provision is worth its weight, because the field is
persisted into every result file and is therefore expensive to reinterpret later.

**The gap: verification and comparability are two different guarantees, and OS-5 as
scoped only delivers the first.**

`suite_version` is written by `runner.py:_load_suite` and persisted onto
`BenchmarkResult`. Confirmed by repository-wide grep: **nothing reads it.**
`storage.py:latest_per_model()` selects the newest result per model regardless of
suite version, and `advisor.py:rank()` accepts whatever it is handed. So after OS-5 as
currently scoped, the server verifies the hash at load and then still lets the advisor
rank a result produced under suite A against one produced under suite B.

That is verification with no consumer — decorative in the same structural way the
literal was decorative. The plan's own framing ("`suite_version` verifies something")
is satisfied by load-time checking alone, but the milestone's *purpose*
(reproducibility) is not.

Two defensible closures. Pick one explicitly; do not leave it implied:

- **(a) Close it in OS-5.** Add a consumption-side rule: `latest_per_model()` groups by
  `(model_name, suite_version)` or the advisor warns on mixed suite versions.
  Recommended location is **`storage.py`, not `advisor.py`** — see §4.
- **(b) Defer it, with the reason recorded.** "OS-5 establishes suite identity; using
  identity as a comparability key is a separate concern." Legitimate, but then the ADR
  must say so, or the next reader assumes it was overlooked.

Recommendation: **(a)**, in `storage.py`. It is roughly fifteen lines, it is the
difference between a verified field and a *used* verified field, and the 25 stored
results all share one suite version so it is a no-op on existing data.

**Comparability is broader than suite identity.** A stored result is comparable to
another only if suite, hardware, `max_model_len`, and `concurrency` all agree. Current
state:

| Dimension | Enforced? |
|---|---|
| `hardware` | Yes — `advisor._hardware_matches` (DEC-036, exact GPU name + 0.5 GB VRAM tolerance) |
| `max_model_len` | Warned — `_maybe_warn_max_model_len` |
| `suite_version` | **No** — the OS-5 gap above |
| `concurrency` | **No** — recorded on `BenchmarkResult`, read by nothing |
| Runtime/engine version | **No** — not recorded at all |

`concurrency` is the second unchecked axis. All 25 stored results are `concurrency: 1`,
so it is latent today; it becomes a live wrong-answer generator the first time anyone
runs the suite at a different concurrency. It costs one clause in the same grouping
key as `suite_version`.

Runtime identity is deliberately **out of scope**. `suite_version` pins the *input* to
the benchmark; it says nothing about the engine that consumed it. Two results with an
identical suite hash and identical hardware are still incomparable if produced by
different engine versions or server settings. That is the manifest's job, and the
manifest is not Phase A. The obligation on OS-5 is narrower but real: state in the ADR
that `suite_version` is a **necessary but not sufficient** comparability key, so that
the field is not later mistaken for a full provenance token, and choose a format
(algorithm tag) that does not foreclose sitting alongside other keys.

**Failure modes.**

1. *Mismatch at load* — a prompt was edited without regenerating. Plan says raise
   loudly. Correct: this is a first-party developer-facing tool and a silent wrong
   number is worse than a stopped run. Contrast with DEC-047 §4's fail-open posture
   for *admission* — that is a serving path with a live client; this is a benchmark
   harness. Different posture, deliberately. Record the contrast in the ADR so the
   raise is not later "corrected" toward fail-open by someone applying DEC-047 out of
   context.
2. *Missing `suite_version` key* — a hand-written custom suite. Must raise with a
   message naming `make suite-version`, not `KeyError`.
3. *Malformed suite JSON* — already raises via `json.loads`. Fine.
4. **Silent result loss.** `storage.py:all_results()` wraps parse and validate in
   `except Exception: continue`. An unparseable or schema-invalid result file is
   dropped with no signal. This is the archival failure mode that matters: a corrupt
   file makes the advisor rank a *smaller* set and nothing says so. In scope for OS-5
   as the archival-integrity item — the minimum fix is to count what was skipped and
   surface it in `AdvisorReport.warnings`, which already exists as a channel.
5. *Regeneration command drift* — `make suite-version` computing a hash by a different
   code path than `_load_suite` verifies. Both must call one function. One
   implementation, two callers.

**Archival strategy.** Files accumulate in `benchmarks/results/` forever (25 today, all
one suite version, 7 distinct models). No pruning, no index, filename-encoded
timestamps. This is adequate and should not be replaced. The correct minimum is:
results are **immutable and append-only**; incomparable results are *excluded by the
consumer*, never deleted. Deleting history to make the advisor tidy is the wrong
instinct — the whole point of a verified suite hash is that you can tell which stored
results still count. Nothing more than the grouping key in (a) is needed.

**Acceptance criteria for OS-5.**

1. `sha256` of the canonical form of the shipped suite equals the literal `b47066…`
   *(this is the migration proof)*.
2. Mismatch between computed and stored raises, with a message naming the regeneration
   command.
3. A missing `suite_version` key raises the same actionable error, not `KeyError`.
4. `make suite-version` and `_load_suite` verification call the same function.
5. Canonicalization is order-, whitespace-, and escape-independent: a suite that
   differs only in key order, indentation, or `\uXXXX` spelling hashes identically.
6. A suite differing in any prompt `text` or `label`, or in prompt order, hashes
   differently.
7. All 25 existing stored results still load and still rank *(no migration occurred)*.
8. If closure (a) is taken: results from a different suite version are not silently
   ranked against the current one.
9. Unparseable result files are counted and surfaced, not silently dropped.

### 1.2 OS-6 — Advisor honesty

**`quant_score`.** A constant `1.0` at weight `0.10`. Deleting it and renormalising the
remaining weights to `0.40/0.30/0.20 ÷ 0.9` = `0.444/0.333/0.222` is correct and the
consequence is provable:

```
old score = 100 · S + 10        where S = 0.40·t + 0.30·f + 0.20·v
new score = (100/0.9) · S
```

Both are strictly increasing affine functions of the same `S`. **Ranking among viable
models is invariant.** That is the acceptance criterion: assert the ranked order of
viable models is unchanged across the weight change, for a fixture with ≥3 models.
(Modulo `round(score, 2)` ties, which the test fixture should avoid.)

**The one behavioral change is the floor collapse, and it is not obvious.** Under the
old formula every viable model scored at least `10.0`, because the constant
contributed `+10` unconditionally. Non-viable models are hard-set to `0.0`. So
`score > 0` was an accidental but reliable proxy for `viable`. After the change, a
viable model with `S = 0` scores `0.0` — indistinguishable from a gated one by score
alone. `S = 0` requires `throughput_score = 0`, i.e. a zero-throughput result: a failed
or empty benchmark run. Rare, not impossible.

The remedy is not code, it is contract: **`viable` is the only viability signal;
`score` must never be used as one.** `AdvisorResult` already carries `viable`. Add the
statement to the ADR, and a test that a zero-throughput viable model and a
VRAM-gated model are distinguishable via `viable` while both scoring `0.0`. Also note
`advisor_results.sort(key=..., reverse=True)` is stable, so ties at `0.0` fall back to
input order — which is `latest_per_model()` dict order, i.e. arbitrary. Worth one line.

`test_scores_are_normalized_0_to_100` asserts the score range and will need updating.

**The rename is an attribution claim, not a spelling change — and this is the strongest
finding in OS-6.**

```python
def _peak_vram_footprint_gb(before, after):
    min_free = min(before.vram_free_gb, after.vram_free_gb)
    return max(0.0, before.vram_total_gb - min_free)
```

This is `total − min(free)`: **whole-device occupancy**, not this model's footprint. It
includes the desktop compositor, every other process on the GPU, and any other model
resident in the pool. DEC-032 already conceded the naming when it introduced the
function.

Renaming it to `vram_footprint_gib` fixes the unit and *entrenches the wrong
attribution* — it makes a device-level number sound like a per-model measurement, which
is more misleading than `peak_vram_delta_gb`, a name so obviously wrong that nobody
trusts it. The evidence is in the stored data: `opt-125m`, a 125M-parameter model,
records `7.09 GB` in one run and `1.65 GB` in another on the same RTX 4060. The
difference is what else was on the GPU, not what the model needed.

Recommend `vram_device_occupied_gib` (or `gpu_vram_in_use_gib`) — a name that survives
someone reading it literally. If the plan's `vram_footprint_gib` is kept, the ADR must
state in one sentence that the value is device occupancy and is an upper bound on the
model's footprint, so the name and the record disagree in writing rather than silently.

This propagates to `_check_vram_budget`, which compares this device-occupancy number
against `estimate_engine_footprint_gib` — a *per-model* estimate. Comparing occupancy
to footprint systematically over-reports "budget exceeded" on any machine with a
display attached. Pre-existing, not introduced by OS-6, and out of scope to fix — but
it must be named, because OS-6 is the change that touches every one of these call
sites and "we were in the file and said nothing" is how a defect becomes permanent.

**Unit consistency.** Verified: `hardware.py` divides by `1024³` (NVML path) and MiB by
`1024` (nvidia-smi path). Every VRAM number in the system is already GiB. So the
`_gb → _gib` rename is a *correction of the label*, and stored values are numerically
unchanged — the rename is safe.

But OS-6 as scoped renames one field and leaves `HardwareProfile.vram_total_gb` and
`vram_free_gb` — also GiB — mislabeled. The result is a codebase where
`result.vram_footprint_gib` is compared against `hardware.vram_total_gb` in the same
expression, both GiB, differently spelled. That is worse than uniform wrongness,
because it implies a conversion that isn't there.

Recommend extending the rename to both `HardwareProfile` fields. Cost, stated honestly:
it pulls `hardware.py`, `test_hardware.py`, `test_advisor.py` fixtures,
`test_benchmark_runner.py:_hw`, and the `/v1/benchmark/results` + `/v1/benchmark/advise`
response bodies into OS-6, and needs the same compatibility alias treatment as
`BenchmarkResult`. That is a real scope increase and the call is the user's — but
"rename one of three identically-wrong fields" should be a decision, not an accident.

**Recommendation weights.** `40/30/20` (renormalised `44/33/22`) are unjustified round
numbers. They are also *fine*: no data exists to calibrate them against, and inventing
a calibration procedure is not Phase A. The honest position is that they are a declared
editorial preference, not an inference. Record that in the ADR and stop. What is **not**
acceptable is the current docstring presenting them as reasoned ("most visible
performance signal") without saying they are uncalibrated.

**The normalization is relative, and the output does not say so.** This is the deepest
issue in the advisor and it is *out of scope* to fix:

- `throughput_score = throughput / max_throughput` and
  `ttft_score = 1 − ttft / max_ttft` normalise against *the models present in this
  report*. Adding a model changes every other model's score.
- The worst-TTFT model always scores exactly `0` on that component. With a single
  result, `ttft_score = 0` always — so **one model can never score above ~74** under the
  new weights (`0.444 · 1 + 0.222 · v`), for no reason connected to the model.
- If all models have similar TTFT (100 ms vs 101 ms), the 30% weight amplifies a 1%
  real difference across the full component range. The normalization manufactures
  signal from noise.

`score` is therefore a **within-report ordinal**, not a portable quality metric — but it
is persisted, served over `GET /v1/benchmark/advise`, and formatted as a 0–100 number,
all of which say "absolute" to a reader.

Redesigning normalization is a scope expansion OS-6 should refuse. The in-scope,
honesty-consistent action is to *state the property and pin it with a test*: the
single-result `ttft_score = 0` cap is a concrete, testable instance, and a test that
adding a model changes the scores of the others is documentation that cannot rot.

**Explainability.** `recommendation_str` is a formatted string built at scoring time —
`"{model} — {tps:.0f} tok/s, {ttft:.0f}ms TTFT, {vram:.2f} GB VRAM"`. It reports the
*inputs* and never the *decision*: it does not say which component dominated the score,
and for a gated model it explains the gate but for a ranked model it explains nothing
about the rank. Component scores are computed and discarded. Surfacing them
(`throughput_score`, `ttft_score`, `vram_score` on `AdvisorResult`) would make the score
auditable and is a small additive change — but it is additive, and Phase A's exit
criterion is that every reported number is *true*, not that more numbers are reported.
Note it; do not do it in OS-6. It is the natural first item for whatever follows.

**Calibration.** There is none, and none is possible within Phase A: no ground truth
exists for "was this recommendation right." The honest handling is negative — say in the
ADR that the weights are uncalibrated and that `score` has no external referent. A
number presented without that caveat is the same class of defect as `quant_score`:
something that looks like a measurement and isn't.

**Future extensibility.** Removing `quant_score` is correct *because* it is a
placeholder for data that does not exist. When quantization data does exist, it enters
as a measured quantity with its own weight — deleting the placeholder does not make
that harder. Deleting it is strictly better than keeping a lie warm.

### 1.3 Hidden coupling between the Advisor and future runtime changes

Stated as what the current code assumes. No proposals.

1. **`_warm_ttft_ms` drops `prompt_results[0]` as a cold-start artifact.** This assumes
   the first request is the only anomalous one and that subsequent requests are
   independent. Under any runtime where TTFT depends on queue occupancy rather than
   cache warmth, "warm" stops meaning what the function's name says.
2. **`concurrency` is recorded and ignored.** `BenchmarkResult.concurrency` exists;
   nothing reads it. `mean_throughput_tps` is a single scalar with no concurrency
   qualifier. In a runtime where throughput is a function of batch occupancy, a scalar
   throughput is underspecified — and the advisor weights it at 44%.
3. **`_peak_vram_footprint_gb` attributes whole-device occupancy to one model.** With
   one model resident this is a loose upper bound. With more than one resident it is
   not attributable at all, and the hard VRAM viability gate is driven by it.
4. **`suite_version` pins the benchmark input, not the runtime that consumed it.**
   Results with identical suite hashes are still runtime-incomparable. This is the
   coupling that argues for keeping the field format extensible (algorithm tag) rather
   than for building anything now.
5. **`_check_vram_budget` compares a measured device occupancy against a static
   analytic estimate.** Two different quantities under one comparison; the estimate's
   assumptions and the measurement's scope must stay in sync by hand.

All five are properties of code that exists today. None requires action in Phase A.
They are listed so that the ADRs record what the advisor is currently assuming, and so
that a later change to any of these is recognised as invalidating advisor output rather
than as an unrelated improvement.

### 1.4 Verdict

Both units remain the correct final Phase A milestones. Neither should be re-scoped.
Two boundary corrections are required before authoring:

- **OS-5:** the migration premise is void (§0); the canonicalization must be pinned;
  and the verification-without-a-consumer gap must be closed or explicitly deferred.
- **OS-6:** the rename target must be re-decided (device occupancy, not footprint), and
  the floor collapse must be recorded as a contract about `viable` vs `score`.

**APPROVE WITH MODIFICATIONS.**

---

## 2. Required ADRs

**DEC-054 — Benchmark suite identity is a pinned content hash** *(OS-5)*

- Canonical form: `sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False,
  separators=(",", ":")).encode("utf-8"))`, hashing `data["prompts"]` only — the
  `suite_version` key is excluded because it lives inside the file it identifies.
- Stored algorithm-tagged (`sha256:…`); bare hex accepted for the 25 existing results.
- **Supersedes the plan's migration premise.** The computed hash equals the stored
  literal `b47066…`. Cites and voids plan §457, §808 item 2, §256, §889. All 25 stored
  results remain comparable; nothing is regenerated; there is no rollback hazard.
- Records that the earlier reviews were correct about the code (no `sha256` call
  exists) and that only the inference "therefore the value is arbitrary" is disproved.
- Records the three alternatives rejected: hand-bumped semver (reproduces the current
  defect), git blob hash (VCS-coupled, over-invalidates), signed manifest (wrong threat
  model).
- Records the limitation: byte-identity of canonical JSON; Unicode NFC/NFD spellings of
  the same prompt hash differently. Not normalised — the suite is ASCII.
- Records the fail-loud posture and *why it does not contradict DEC-047 §4*: DEC-047's
  fail-open applies to the serving admission path with a live client; this is a
  first-party benchmark harness where a silent wrong number is worse than a stopped run.

**DEC-055 — `suite_version` is a necessary but not sufficient comparability key**
*(OS-5)*

- Identity of the benchmark *input*, not of the runtime that consumed it.
- The full comparability tuple today is `(suite_version, hardware, max_model_len,
  concurrency)`; hardware is enforced (DEC-036), `max_model_len` warns, the other two
  are unchecked.
- Records whether closure (a) or (b) was taken, and that `concurrency` is the second
  unchecked axis (latent: all 25 stored results are `concurrency: 1`).
- Records that results are immutable and append-only: incomparable results are excluded
  by the consumer, never deleted.

*May be folded into DEC-054 if the OpenSpec author prefers one record. Keep them
separate if closure (b) is taken, since the deferral then needs its own rationale.*

**DEC-056 — Advisor score reflects measured quantities only** *(OS-6)*

- `quant_score` deleted; weights renormalised to `0.444 / 0.333 / 0.222`.
- Records the affine-invariance proof: old `100·S + 10`, new `(100/0.9)·S`, both
  strictly increasing in `S` ⇒ ranking among viable models is unchanged.
- **Records the floor collapse:** the constant was an accidental floor separating
  "viable but worst" (`10.0`) from "non-viable" (`0.0`). It is gone. **`viable` is the
  only viability signal; `score > 0` must never be used as a proxy.**
- Records that ties at `0.0` sort by arbitrary input order.
- Records that the weights are an uncalibrated editorial preference with no external
  referent, and that `score` is a **within-report ordinal** — relative normalization
  means adding a model changes every other model's score, and a single-result report
  caps at `ttft_score = 0`.

**DEC-057 — The VRAM number is device occupancy, not model footprint** *(OS-6)*

- `total − min(free)` measures whole-device VRAM occupancy including other processes
  and other resident models. Evidence: `opt-125m` recorded 7.09 GB and 1.65 GB on the
  same RTX 4060 in different runs.
- Records the chosen name and *why*. If `vram_footprint_gib` is kept over
  `vram_device_occupied_gib`, this ADR is where the name and the measurement are
  reconciled in writing.
- Records the unit correction: every VRAM value in the system is already GiB
  (`hardware.py` divides by `1024³` / MiB÷1024), so `_gb → _gib` changes the label and
  not the number — the rename is numerically safe.
- Records the decision on `HardwareProfile.vram_total_gb` / `vram_free_gb` — renamed
  with OS-6, or knowingly left inconsistent.
- Records, without fixing, that `_check_vram_budget` compares this occupancy against a
  per-model analytic estimate and therefore over-reports on any machine with a display.
- Records the advisor's runtime assumptions (§1.3 items 1–5) as known coupling.

---

## 3. Required OpenSpec refinements

### OS-5 — `add-verified-benchmark-suite-identity`

**Add:**
- A requirement pinning the canonical form verbatim, with scenarios for
  order-independence, whitespace-independence, escape-independence, and
  content-sensitivity.
- A requirement that computed ≠ stored raises with a message naming the regeneration
  command; and that a missing `suite_version` key raises the same actionable error.
- A requirement that `make suite-version` and load-time verification share one
  implementation.
- **A requirement that the shipped suite's computed hash equals `b47066…`** — the
  migration proof.
- If closure (a): a requirement that results carrying a different `suite_version` are
  not silently ranked against the current suite.
- A requirement that unparseable result files are counted and surfaced in
  `AdvisorReport.warnings` rather than silently skipped.

**Delete before it is written** (each traceable to §0):
- Any migration task. There is no migration.
- Any rollback-hazard task for the regenerated hash (plan §808 item 2).
- Any task regenerating `benchmarks/prompts/standard.json` (plan §457) — the file is
  read-only in this change.
- Any spec text asserting that stored results become incomparable (plan §256, §889).

**Explicit non-goals for tasks.md:** no runtime/engine version in the suite hash; no
Unicode normalization; no manifest work; no result pruning or archival tooling; no
changes to `advisor.py` (see §4); no changes to the prompt suite contents.

### OS-6 — `add-honest-advisor-scoring`

**Add:**
- A requirement that the score is composed only of measured quantities, with a scenario
  asserting **ranked order of viable models is unchanged** across the weight change.
- A requirement that `viable` is the sole viability signal, with a scenario in which a
  zero-throughput viable model and a VRAM-gated model both score `0.0` and remain
  distinguishable.
- A requirement fixing the persisted field name + compatibility alias, with a scenario
  loading a stored result carrying the old key.
- A scenario pinning the within-report ordinal property: a single-result report has
  `ttft_score = 0`, and adding a model changes the others' scores.

**Decide before authoring** (both are user calls, not author calls):
- Rename target: `vram_footprint_gib` as planned, or `vram_device_occupied_gib`.
- Whether `HardwareProfile.vram_total_gb` / `vram_free_gb` are renamed in the same
  change.

**Explicit non-goals:** no change to the normalization scheme; no component-score
surfacing on `AdvisorResult`; no calibration procedure; no change to
`_check_vram_budget`'s comparison; no change to `_warm_ttft_ms`; no concurrency-aware
throughput.

---

## 4. Repository impact

### OS-5

| Path | Change | Note |
|---|---|---|
| `benchmarks/runner.py` | `_load_suite` computes and verifies | OS-5 owns suite load/hash only |
| `benchmarks/prompts/standard.json` | **read-only** | value already correct |
| `benchmarks/storage.py` | closure (a): group by `(model, suite_version)`; count skipped files | **new touch point, not in plan §5.1** |
| `Makefile` | `make suite-version` | calls the shared function |
| `scripts/` | regeneration entry point if not pure-Make | |
| `tests/unit/test_benchmark_runner.py` | canonicalization + mismatch tests | |
| `tests/unit/test_storage.py` | grouping + skipped-count tests | closure (a) only |
| `docs/DECISIONS.md` | DEC-054, DEC-055 | |

**`storage.py` is a deviation from the plan and it is deliberate.** The instinct is to
put the comparability filter in `advisor.py` — that is where ranking happens. Do not.
`advisor.py` is OS-6's file. Touching it from OS-5 creates a shared-file contention
that the plan explicitly claims does not exist (§4.1 declares OS-5 parallel with
OS-2/OS-3; §7 lists OS-2/OS-5 as the *only* contention pair). `latest_per_model()` is
already the function that decides which results reach the advisor, so the filter
belongs there on the merits, and keeping it there preserves the parallelism the plan
depends on. **`benchmarks/advisor.py` is out of bounds for OS-5.**

The plan's §5.1 touch-point table should gain `benchmarks/storage.py` under OS-5.

### OS-6

| Path | Change | Note |
|---|---|---|
| `benchmarks/advisor.py` | delete `quant_score`, renormalise, rename reads | |
| `benchmarks/schemas.py` | rename + compatibility alias on `BenchmarkResult` | persisted field |
| `benchmarks/runner.py` | rename at construction + `_peak_vram_footprint_gb` | shared with OS-2 — the known contention |
| `benchmarks/hardware.py` | *conditional* — only if the `HardwareProfile` rename is taken | |
| `api/routes/benchmark.py` | response bodies change if `HardwareProfile` is renamed | otherwise untouched |
| `tests/unit/test_advisor.py` | `test_scores_are_normalized_0_to_100` needs updating; new invariance + floor tests | 19 tests today |
| `tests/unit/test_benchmark_runner.py` | `_hw` fixture if the `HardwareProfile` rename is taken | |
| `tests/unit/test_hardware.py` | same condition | |
| `benchmarks/results/*.json` | **read-only** — 25 files exercise the alias | |
| `docs/DECISIONS.md` | DEC-056, DEC-057 | |

The 25 stored files are the compatibility alias's real test corpus. `all_results()` must
load all 25 after the rename; that is the acceptance check, not a synthetic fixture.

---

## 5. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Implementer picks a canonicalization other than the one already in the file and manufactures a break that need not happen | **High** | DEC-054 pins the form verbatim; the acceptance test asserts equality with `b47066…`. This replaces the plan's stated risk, which was the inverse. |
| R2 | OS-5 ships load-time verification with no consumer — a verified field nothing reads | **High** | Closure (a) or an explicit (b) with recorded reason. Not left implied. |
| R3 | `vram_footprint_gib` makes a device-occupancy number sound like a per-model measurement — more misleading than the name it replaces | **High** | Re-decide the name, or reconcile name and measurement in DEC-057. |
| R4 | OS-5's comparability filter lands in `advisor.py`, breaking the plan's parallelism claim | Medium | `storage.py` only; `advisor.py` out of bounds for OS-5 (§4). |
| R5 | The floor collapse silently breaks a consumer using `score > 0` as viability | Medium | DEC-056 states the contract; test asserts both cases score `0.0` and stay distinguishable. Internal consumers are `scripts/advise.py` and `/v1/benchmark/advise` — audit both. |
| R6 | Renaming one of three identically-mislabeled GiB fields leaves a codebase implying a conversion that does not exist | Medium | Decide explicitly; do not let it happen by omission. |
| R7 | Partial `HardwareProfile` rename changes `/v1/benchmark/*` response bodies without an alias | Medium | If taken, alias `HardwareProfile` the same way as `BenchmarkResult`. |
| R8 | OS-2/OS-6 contention on `PromptResult` construction in `runner.py` | Medium | Plan's existing sequencing edge (OS-6 after OS-2) already handles it. Unchanged. |
| R9 | `score` continues to be read as an absolute metric across reports | Low–Med | Documented + pinned by test. Not fixed — out of scope by design. |
| R10 | The fail-loud raise is later "corrected" toward fail-open by someone applying DEC-047 §4 | Low | DEC-054 records why the postures differ. |

---

## 6. Recommended implementation order

The plan's sequencing (`OS-5 ∥ OS-2/OS-3`, `OS-6` after `OS-2`) is correct and unchanged
by this review. Within the two units:

**Pre-work — two decisions before either OpenSpec is authored.** Both are the user's,
both change what gets written, and neither has a defensible default:

1. OS-5 closure (a) — filter in `storage.py` — or (b) — defer with reason.
2. OS-6 rename target, and whether `HardwareProfile` is included.

**OS-5** *(depends on OS-1; parallel with OS-2/OS-3)*

1. DEC-054 first — the canonicalization is the one irreversible decision, since the
   chosen form is persisted into every future result.
2. The equality test against `b47066…`, before any production code. It is the proof the
   premise correction is real; if it fails, stop, because the canonicalization is wrong.
3. Shared hash function + `_load_suite` verification + missing-key error.
4. `make suite-version` calling that same function.
5. Closure (a): `storage.py` grouping and the skipped-file count. **Last**, so that
   1–4 remain a coherent, independently revertible unit if (a) is dropped.
6. DEC-055.

**OS-6** *(after OS-2 merges; parallel with OS-4 in the plan's original ordering)*

1. DEC-056 + DEC-057 first — the rename target and the `viable`/`score` contract are
   both decisions, and doing them after the code means the code decides.
2. The ranking-invariance test, written against the **current** code so it passes
   before the change. A regression test authored after the change proves nothing about
   the change.
3. Delete `quant_score`, renormalise. Invariance test must still pass.
4. Floor-collapse test + `test_scores_are_normalized_0_to_100` update.
5. Rename + compatibility alias. Verify all 25 stored files still load.
6. `HardwareProfile` rename, if taken. Last — it is the only step touching the API
   surface, and isolating it keeps it revertible on its own.

Sequencing rationale in one line: in both units the ADR comes first because the
irreversible content is a *decision*, and the characterisation test comes before the
code change because a test written afterwards only records what the code does.
