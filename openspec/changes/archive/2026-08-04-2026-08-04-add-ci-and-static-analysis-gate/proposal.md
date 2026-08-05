# OS-1 — CI and static-analysis gate

- Change ID: `2026-08-04-add-ci-and-static-analysis-gate`
- Milestone: Phase A, unit OS-1 (`docs/PHASE-A-EXECUTION-PLAN.md` §4)
- Traces to: review task A4
- Depends on: nothing
- Blocks: OS-2, OS-3, OS-4, OS-5, OS-6

## Summary

Add an automated quality gate to the repository: a GitHub Actions workflow that
runs the unit suite, a linter, and a type checker on every pull request and every
push to the long-lived branches, plus the tool configuration and dev dependencies
that make those checks runnable.

This change adds no behaviour. No file under `src/` or `tests/` is modified. It is
the first merge-blocking gate in the repository's history, and it exists so that
every subsequent Phase A change can be verified by machine rather than by the
author's assertion.

## Motivation

The repository has 430 unit tests across 37 files and no `.github/workflows/`
directory. The suite runs only when a human remembers to run it, and the only
evidence a reviewer has that a change is safe is the author's claim.

Phase A makes that gap acute rather than merely untidy. Five of the six Phase A
units edit files that other units also edit — `services/chat_service.py`,
`benchmarks/runner.py`, `routing/admission.py`, `schemas/chat.py` — and several
branches will be in flight simultaneously (`docs/PHASE-A-EXECUTION-PLAN.md` §7).
Unverified claims do not compose: with five concurrent branches over a shared file
set, "I ran the tests" describes a state that no longer exists by the time the
branch merges.

There is a second reason specific to this milestone. OS-2 changes the shape of
data flowing through `observability/middleware.py`, whose failure mode is silent —
a wrong metric does not raise. The tests are the only detector, and a detector
nobody runs is not a detector. Phase A is a milestone about the difference between
a reported number and a true one; shipping it without automated verification would
be the same category of error the milestone exists to correct.

Sequencing rationale is settled in `docs/PHASE-A-EXECUTION-PLAN.md` §6.2 and is
not reopened here.

## Scope

1. **Continuous integration workflow.** A GitHub Actions workflow that, on pull
   requests targeting and pushes to the long-lived branches:
   - installs dependencies via `uv sync` on the Python version declared in
     `pyproject.toml`;
   - runs the unit suite (`tests/unit`);
   - runs the linter in check mode;
   - runs the type checker over `src/`.

   The job runs on a standard Linux runner with no GPU. The suite is already
   GPU-free by design; this change does not alter that property, and no GPU job is
   introduced.

2. **Tool configuration in `pyproject.toml`.** Linter and type-checker
   configuration sections, colocated with the existing `[tool.pytest.ini_options]`
   so that tool policy has one home.

3. **Dev dependencies.** The linter and type checker added to the existing dev
   dependency group. Neither is currently installed, so this change adds
   dependencies as well as configuration.

4. **A type-check baseline.** An enumerated, per-module suppression list covering
   the modules that interface with untyped third-party surfaces (vLLM, CUDA, NVML).
   The baseline is subject to three rules:
   - it is a **closed, explicit list of modules** — no repository-wide suppression,
     and no directory-level wildcard that would silently absorb new files;
   - it is derived from the initial type-checker run against the tree as it stands,
     not from a guess made in advance;
   - it is written once, here, and **does not grow during OS-2 through OS-6**.
     Baseline growth in a later Phase A unit is a review signal that the unit is
     touching more than it should (`docs/PHASE-A-EXECUTION-PLAN.md` §7.4).

5. **Lint rule selection constrained by the existing tree.** The initial rule set
   is chosen so that the repository passes **without source edits**. Where a
   desirable rule would require modifying existing code, it is either omitted from
   the initial selection or recorded as a scoped per-path exclusion. Fixing the
   code is a separate concern and belongs to a separate change.

6. **Contributor documentation.** The lint and type commands added to the local
   setup instructions and the pull-request checklist in `CONTRIBUTING.md`, so that
   contributors can reproduce the gate locally before pushing.

7. **One decision record.** An ADR in `docs/DECISIONS.md` recording the baseline
   policy: that a type-check baseline is accepted in preference to clean types,
   which modules it covers, and that it must not grow within Phase A. Claim the
   next free DEC number when opening the pull request
   (`docs/PHASE-A-EXECUTION-PLAN.md` §7.3); do not pre-assign it in this proposal.

8. **Branch protection.** Enabling the workflow as a required status check on the
   long-lived branches. This is a repository-settings action performed by a
   maintainer, not a file in the diff — see Acceptance Criteria and Risks.

9. **Prerequisite: stop ignoring `.github/`.** Discovered during implementation.
   `.gitignore` excluded the directory, making a tracked workflow impossible. The
   whole line is removed rather than replaced with a narrower negation, for three
   reasons:
   - **The obvious narrow form does not work.** Git cannot re-include a path whose
     parent directory is excluded, so `.github/` plus `!.github/workflows/` matches
     nothing at all. Only the contents form (`.github/*` plus a negation)
     functions — a distinction that reads as a typo and silently breaks later.
   - **A negation encodes a claim the repository does not hold.** `.github/` is
     GitHub's designated location for repository metadata. Excluding it by default
     means every legitimate addition — issue templates, a PR template, `CODEOWNERS`,
     `dependabot.yml` — must re-argue the exclusion. `github-templates.md` at the
     repository root is the existing cost of that: templates whose stated
     destination is `.github/ISSUE_TEMPLATE/` are staged as a root markdown file
     because the directory was unavailable.
   - **Nothing there needs hiding.** `.github/` has never been tracked in this
     repository's history and contains exactly one file: the workflow this change
     adds.

   This is a repository correction that unblocks the approved scope. It adds no
   capability and changes no behaviour. OS-1 introduces **no `.github/` content
   beyond `workflows/ci.yml`**; populating the templates documented in
   `github-templates.md` is separate work and is not done here.

## Explicit non-goals

- **No source or test changes.** No file under `src/` or `tests/` is modified.
  This is the single most important boundary in this change. If landing the gate
  appears to require editing source, the correct response is to relax the rule
  selection, not to edit the source.
- **No automatic formatting.** Formatting is not run in CI and no formatter pass is
  applied to the tree. A repository-wide reformat would produce a diff that makes
  every subsequent Phase A change unreviewable.
- **No clean type-check run.** A fully typed `src/` is not a goal of this change or
  of Phase A. The baseline is the deliverable.
- **No coverage measurement, threshold, or reporting.**
- **No integration or contract test tier.** The absence of one is a real gap
  (`docs/PHASE-A-EXECUTION-PLAN.md` §5.2) and remains out of scope.
- **No GPU job, self-hosted runner, or hardware-dependent check.**
- **No release, publish, or artifact workflow.**
- **No dependency-audit or dependency-update automation.** `pip-audit` is already a
  dev dependency; automating it is a separate concern.
- **No behavioural change of any kind.** No endpoint, schema, engine, or
  configuration semantics are touched.
- **No change to the Engine Boundary.** OS-1 does not touch `engines/`. The Phase A
  Engine Boundary change is OS-2's, and its conformance argument is settled by
  DEC-047 and recorded in `docs/PHASE-A-EXECUTION-PLAN.md` §1.1.
- **No work belonging to other Phase A units.** Token counting (OS-2), `seed`
  (OS-3), `resolved`/`warnings` (OS-4), `suite_version` (OS-5), advisor scoring
  (OS-6), and archiving settled changes (HK-1) are all out of scope.
- **No relaxation of `vllm` as a required dependency.** See Risks.

## Capabilities

### New capabilities

(none)

### Modified capabilities

(none — this change adds no runtime capability. The spec delta under `specs/`
records one platform-level requirement about automated verification of changes,
which is a genuine property of the platform's development process and satisfies
`openspec validate`'s requirement that every change carry at least one delta.)

## Repository impact

**Added**

- `.github/workflows/` — new top-level directory. Owns merge admissibility. This
  is the first enforcement authority in the repository.

**Modified**

- `pyproject.toml` — gains linter and type-checker configuration and two dev
  dependencies. Becomes the single source of lint and type policy.
- `uv.lock` — regenerated as a consequence of the added dev dependencies.
- `.gitignore` — one line removed (`.github/`). **Prerequisite correction, not a
  feature.** The repository ignored its own metadata directory, so a tracked
  workflow was impossible and this change would have delivered an inert file. The
  line sat in the top block beside `.venv/` and `.idea/` with no rationale comment
  — unlike `article-final.md` and `benchmarks/results/`, which both carry one —
  `.github/` had never been tracked, and `github-templates.md` at the repository
  root exists as a workaround for the very exclusion being removed. Removing the
  whole line rather than negating a subpath is explained in Scope item 9.
- `CONTRIBUTING.md` — setup instructions and PR checklist gain the two new
  commands. **Overlaps with HK-1**, which corrects the stale test count in the same
  file; see Dependencies.
- `docs/DECISIONS.md` — one appended ADR.

**Not modified**

- Everything under `src/`.
- Everything under `tests/`.
- Everything under `config/`, `openspec/specs/`, `playground/`, and
  `benchmarks/prompts/`.

**Ownership note.** `.github/workflows/` owns *whether a change may merge*. It does
not own what the code does. This change introduces no new module boundary and does
not alter the dependency direction described in `docs/ARCHITECTURE.md`.

## Acceptance criteria

Phase A acceptance criteria A.1 and A.2 are satisfied by this change and are not
re-verified by later units.

### Infrastructure complete

1. A pull request whose branch contains a failing unit test does not show a
   passing check.
2. A pull request whose branch contains a lint violation under the selected rule
   set does not show a passing check.
3. A pull request whose branch introduces a type error in a module **outside** the
   baseline does not show a passing check.
4. The workflow is green on the default development branch at merge time, with
   **zero files changed under `src/` or `tests/`** in this change.
5. The linter and type checker are both resolvable after a clean `uv sync`; no
   contributor needs to install a tool by hand to reproduce the gate.
6. The type-check baseline is an enumerated list of specific modules. There is no
   repository-wide suppression and no wildcard that would absorb files added
   later.
7. The workflow requires no GPU and runs to completion on a standard hosted Linux
   runner.
8. Negative validation completed: deliberate test failure, lint violation, and
   out-of-baseline type error each produce a red check (see Validation step 2).

### Repository integration complete

9. The workflow is configured as a **required status check** on the long-lived
   branches, so that a red run blocks merge rather than merely reporting. A
   workflow file alone does not satisfy Phase A acceptance criterion A.1
   (`docs/PHASE-A-EXECUTION-PLAN.md` §9) — the setting must be verified as enabled.
10. The baseline policy — including the statement that the baseline does not grow
    during Phase A — is recorded in `docs/DECISIONS.md`.
11. `CONTRIBUTING.md` documents the lint and type commands in setup instructions
    and the pull-request checklist so contributors can reproduce the gate locally.

*Artifact precondition (not a delivery criterion):* `openspec validate --strict`
passes for this change. It was green when these artifacts were authored; task 3.7
re-runs it after the documentation edits in section 4, since those change files
the validator reads.

## Validation

1. **Local reproduction.** From a clean checkout: `uv sync`, then the unit suite,
   the linter, and the type checker. All three pass with no source modifications.
2. **Negative test of the gate — required** (Infrastructure complete criterion 8).
   On a scratch branch, introduce, one at a time and reverting each before the
   next:
   - a deliberately failing assertion in an existing test;
   - a deliberate lint violation;
   - a deliberate type error in a module outside the baseline.

   Confirm the check goes red for each. Delete the scratch branch. **A gate that
   has only ever been observed green has not been validated** — this step is the
   substance of the change's verification, not a formality.
3. **Positive confirmation.** Confirm the check is green on the unmodified
   development branch, and that the run required no GPU.
4. **Required-check confirmation.** Confirm on a throwaway pull request that a red
   run blocks the merge button, not merely that it reports a failure.
5. **Baseline inspection.** Confirm by reading the configuration that every
   suppressed module is named individually, and record the list in the ADR.
6. **Runtime observation.** Record the wall-clock duration of a cold run. This is
   observation, not a threshold to enforce; it is captured because it feeds the
   dependency-installation risk below.

Per `docs/OPENSPEC_SETUP.md`, a change that cannot be validated is not complete.
Step 2 is what makes this change validatable at all.

## Rollback strategy

Rollback is complete and consequence-free:

- Delete the workflow file.
- Revert the `pyproject.toml` and `uv.lock` changes.
- Revert the `CONTRIBUTING.md` addition.

No behaviour is restored because none was changed. No data shape, stored artifact,
or published number depends on this change, so nothing becomes incomparable — this
is the only Phase A unit with that property.

The ADR is the one exception. If the change is reverted, the ADR stays and gains a
superseding note explaining why, rather than being deleted. A decision that was
made and then unmade is part of the record.

Partial rollback is also safe: the configuration can remain with the workflow
removed (contributors keep local tooling, the gate stops enforcing), or the
baseline can be widened without touching the workflow. Neither leaves a broken
state.

## Risks

**1. The type-check baseline stalls the change indefinitely.** *(Likelihood: high.
Impact: high if mishandled.)* `engines/vllm_engine.py` and
`utils/vllm_pool_config.py` are large modules over untyped third-party surfaces.
If review of this change turns into a negotiation about type coverage, the entire
milestone stalls at its own gate. *Mitigation:* the baseline is explicitly
authorised by `docs/PHASE-A-EXECUTION-PLAN.md` (Rank 2 risk). A clean type-check
run is not a goal of this change and must not become one during its review. Land
the baseline; open a separate issue for reduction.

**2. Lint scope creep into source files.** *(Likelihood: high. Impact: medium.)*
The natural instinct on seeing lint output is to fix it. Doing so here produces a
diff across `src/` that collides with every in-flight Phase A branch and violates
the surgical-change discipline in `CONTRIBUTING.md`. *Mitigation:* non-goal stated
above; Infrastructure complete criterion 4 states zero `src/`/`tests/` edits as a
testable property; rule selection is constrained by what the existing tree already
passes.

**3. Dependency installation cost on the runner.** *(Likelihood: medium. Impact:
medium.)* `uv sync` installs the full dependency set including `vllm` and its
torch stack — multi-gigabyte, and slow on a hosted runner. Cold runs may be long
enough to be irritating, and runner disk limits are a plausible failure.
*Mitigation:* dependency caching, and the recorded runtime from Validation step 6.
**The forbidden mitigation is making `vllm` an optional extra.** DEC-007 keeps it
required, DEC-047 §6 explicitly declines to authorise an optional extra, and
`AGENTS.md` lists it under anti-scope. If installation cost proves genuinely
prohibitive, that is a finding to record and escalate, not a licence to change the
dependency shape inside OS-1.

**4. The workflow exists but does not gate.** *(Likelihood: medium. Impact:
high.)* A workflow file that is not configured as a required status check reports
failures without blocking merges. The change would look complete while delivering
nothing — the most damaging failure mode here, because it is invisible.
*Mitigation:* Repository integration criterion 9 and validation step 4 both target
the setting rather than the file.

**5. `CONTRIBUTING.md` conflict with HK-1.** *(Likelihood: medium. Impact: low.)*
Both changes edit that file. The edits are in different sections and resolve
trivially. *Mitigation:* land HK-1 first (it has no dependencies and is
recommended for day one), or rebase.

**6. The baseline is treated as a standing repository policy.** *(Likelihood: low.
Impact: medium.)* A suppression list is easy to read as permission to leave new
code untyped. *Mitigation:* the ADR states the baseline is a Phase A boundary that
does not grow, not a type-coverage policy.

## Dependencies

**Upstream:** none. This change can begin immediately.

**Downstream:** OS-2, OS-3, OS-4, OS-5, and OS-6 all depend on this change.
Per `docs/PHASE-A-EXECUTION-PLAN.md` §7.4, **no other Phase A branch should be cut
until this change merges** — a branch cut earlier will fail the gate on its first
push for reasons unrelated to its content, and its author will be tempted to fix
lint in files they do not own.

**Non-blocking overlap:** HK-1 (housekeeping) is independent but edits
`CONTRIBUTING.md`. Landing HK-1 first avoids the conflict entirely.

**External:** GitHub Actions must be enabled for the repository, and a maintainer
with administrative access must configure the required status check. Neither is
achievable from within the diff.

**Governance:** `CONTRIBUTING.md` requires one concern per pull request. This
change is one concern — automated verification — and deliberately carries no
behavioural work.

## Future follow-ups

Recorded so they are not smuggled into this change. None is authorised by it.

- **Type-baseline reduction.** Removing modules from the suppression list, one at a
  time. Explicitly not Phase A; open as a standing issue.
- **Integration and contract test tiers.** `tests/` is currently unit-only.
  Introducing a tier that exercises the real engine is a substantial change with
  its own hardware questions.
- **Coverage measurement.** Deliberately omitted here; a threshold introduced
  alongside a new gate tends to be tuned to whatever the tree happens to score.
- **Automated dependency auditing.** `pip-audit` is already a dev dependency.
- **Formatting enforcement.** Requires a tree-wide reformat and should not land
  while Phase A branches are in flight.
- **CI runtime and caching optimisation.** Informed by the runtime recorded in
  Validation step 6.
- **A GPU-dependent job.** Would unlock the token-exact oracle validation discussed
  in the architecture review. Far outside Phase A.
- **No `design.md` accompanies this change.** The only decision of consequence —
  baseline over clean types — is recorded in the ADR and in Scope item 4. If review
  surfaces a genuine design question, add one then.

## Scope Integrity

One prerequisite repository correction was required during implementation:

| Correction | Reason | Why not scope expansion | Why no separate OpenSpec |
|---|---|---|---|
| Removed `.github/` from `.gitignore` | The directory was ignored, so a tracked workflow was impossible; OS-1 would have delivered an inert file | Adds no capability, changes no runtime behaviour, introduces no `.github/` content beyond the approved `workflows/ci.yml` | It only unblocks work already authorised by this change (Scope item 9); splitting it would leave the workflow uncommittable |

No other out-of-plan edits occurred. In particular: no `src/` or `tests/` changes; no optional-`vllm`; no Engine Boundary work; no AsyncLLM work; no opportunistic cleanup under `src/`.

Workflow `concurrency` and `timeout-minutes` are judgement calls targeting proposal Risk 3 (CI cost), not new capabilities.

Remaining incomplete tasks (3.2, 3.4, 3.5, 3.6, 5.1 partial, 5.2) require a push, maintainer branch protection, or post-merge archive — not further engineering inside this change.
