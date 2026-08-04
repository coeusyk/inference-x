# Design — OS-3 Deterministic generation

## Context

Phase A OS-3 (`docs/PHASE-A-EXECUTION-PLAN.md` §4). Today
`ChatCompletionRequest` has no `seed`; clients that send one experience silent
drop. `VLLMEngine._sampling_params` builds vLLM `SamplingParams` without seed.
OS-2 widened streaming for truthful tokens; Engine Boundary must not change
again in Phase A. DEC-047 forbids `execution/` packages and second backends.

Normative contract, ownership, Compatibility Invariants (C1–C15), and B1 live in
`proposal.md`. This design records only the mechanical how.

## Goals / Non-Goals

**Goals:**

- Accept optional request `seed` and forward it unchanged into live
  `SamplingParams` when set.
- Preserve omit-seed ≡ pre-OS-3 kwargs.
- Prove forwarding and sequential identity; mark concurrency as `xfail` (Phase C3).
- Record DEC-051.

**Non-Goals:**

- Response echo / `resolved` / `warnings` / `strict` (OS-4).
- Batch invariance / `deterministic: true` (Phase C3).
- Replay manifests, oracle suite, AsyncLLM, factory, benchmark seed pinning.
- Normalizing backend sentinel semantics (including `-1`).

## Decisions

### D1 — Wire field on `ChatCompletionRequest`, not Engine Boundary

**Choice:** Append `seed: Optional[int] = None` after `stream_options` in
`schemas/chat.py`. Thread only inside `VLLMEngine._sampling_params`.

**Why:** DEC-047 / Phase A §1.1 — Phase A’s Engine Boundary change was OS-2 only.
Seed is wire vocabulary; the concrete backend owns translation to native calls.

**Rejected:** Adding `seed` to `BaseEngine` or a sampling DTO outside `schemas/`.

### D2 — Omit key when unset; forward unchanged when set

**Choice:** `if request.seed is not None: kwargs["seed"] = request.seed`.

**Why:** Sparse kwargs match existing style (G3). Forward-unchanged satisfies G2
and R2 (`-1` included; no runtime normalization).

**Rejected:** Mapping `-1` → omit; rejecting negatives with 422; passing
`seed=None` explicitly.

### D3 — One builder for stream and non-stream

**Choice:** Both paths already call `_sampling_params`; change that method only.

**Why:** G4 path parity by construction; no SSE or driver edits (C3, C5).

### D4 — Benchmarks stay throughput-oriented

**Choice:** Do not modify `benchmarks/runner.py`.

**Why:** Deterministic benchmarking is deferred; harnesses own requesting seed
(R5, C8, N6).

### D5 — Dependency wording Enables OS-4

**Choice:** Proposal metadata says Enables OS-4, not Blocks.

**Why:** Soft edge; OS-4 can ship without seed echo (proposal R7).

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Overclaiming determinism in docs/tests | G5; identity only under sequential single-in-flight; concurrency `xfail` → C3 |
| Implementer normalizes `-1` “helpfully” | R2 / N8 — forward unchanged; review against C15 + B1 only |
| Scope creep into OS-4 response surfaces | Explicit non-goals; do-not-touch list in tasks |
| Identity test flaky on GPU/warmup | Follow existing skip patterns; do not weaken sequential requirement |

## Migration Plan

1. Land DEC-051 as `accepted` with the Contract.
2. Schema + `_sampling_params` + tests in one PR.
3. Optional doc one-liner (honoured, not reproducible).
4. Rollback: revert those files; no metric supersession.

## Open Questions

None. All ambiguities are resolved in `proposal.md` R1–R7.
