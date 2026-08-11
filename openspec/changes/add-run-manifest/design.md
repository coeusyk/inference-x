# Design — add-run-manifest (C1)

## Context

Authoritative spec: `docs/REVIEW-2026-08-03-architecture.md` §8.3 (manifest JSON), §8.1/§8.2
(witness-vs-derive boundary), §6.2a (never silently degrade). Planning report:
`docs/PHASE-C-EXECUTION-PLAN.md` (pending; drafted this session). This design records only the
mechanical decisions; the normative contract is in `specs/platform/spec.md`.

## Goals / Non-Goals

**Goals**
- A content-addressed `run_id` and `X-Run-Id` on non-streaming responses.
- A manifest object assembled from already-shipped signals + minimal new provenance.
- Reconcile the seed-echo contract (requested-seed provenance; no effective-seed echo; no
  reproducibility overclaim).
- Reserve `co_batched_request_ids` without populating it.

**Non-Goals (deferred)**
- Manifest delivery beyond `X-Run-Id` (inline body field / `GET /v1/manifest`) — **[OPEN] D5**.
- `weights_sha256` / `hf_revision` completeness and degradation policy — **[OPEN] D4**.
- ITL `p50/p95`; `runtime.batch_invariant` population (C3); `co_batched_request_ids` population
  (C2); plan/doctor (C6); Varex embedding (C5); any streaming manifest surface; cryptographic
  signing (Phase E).

## Decisions

### D1 — `run_id` is a content hash; content-addressing only
`run_id = "sha256:" + hex(hash(canonical_preimage))`. Reuse the canonicalization discipline of
`benchmarks/suite_identity.py` (DEC-054/055) — deterministic field ordering, stable scalar
formatting, deterministic handling of absent optional fields — rather than a second hasher. **No
signature** is produced (locked decision: signing → Phase E). "Signed" in the roadmap is aspirational.

### D2 — `run_id` preimage = configuration identity, excluding speed/observational/environment blocks
Preimage = `{ engine, model, runtime, sampling, request, warnings }`, where `warnings` is projected
onto each warning's stable identity fields (`type`, `code`, `field`) — **not** its free-text
`message`. **Excluded entirely:** `timing`, `batch`, and `hardware`.

- **`timing`** — speed, not semantics; §8.3 excludes it explicitly.
- **`batch`** — its populated fields in C1 are speed/observational (`cold_start`,
  `prefix_cache_hit_tokens`, `max_batch_size_observed`) — they belong with `timing`, not semantics —
  and the one batch field §8.3 calls semantics-determining, `co_batched_request_ids`, is *reserved
  and unpopulated* here, so hashing an empty batch block is meaningless. See **[OPEN] D6** for the
  batch-block boundary once C2 populates `co_batched_request_ids`.
- **`hardware` — deliberately excluded (decided after implementation review; not an oversight).**
  §8.3's literal "everything outside `timing`" would include it; §8.2 requires hardware be "bound to
  the result," which C1 satisfies by carrying it on the manifest's `hardware` block — binding does
  not require hashing. `run_id` denotes **configuration identity** (engine, model, runtime, sampling,
  request shape, degradation), not **execution-environment identity**. Comparing the same
  configuration across *different* hardware is a primary use case this platform exists to serve
  (Phase A: "help the user decide which model is best for their hardware"); folding `hardware` into
  `run_id` would make that comparison structurally impossible — a matching `run_id` would then only
  ever occur on identical machines. So `hardware` travels on the manifest as the provenance a
  consumer reads to judge *whether* a cross-hardware comparison is reproducible-grade or only
  configuration-comparable — that judgment belongs to the consumer (Varex, a human), not to identity.
  This preserves §8.3's comparability/reproducibility distinction rather than weakening it: `run_id`
  equality still asserts nothing about bitwise reproducibility (which independently requires
  identical hardware, per §8.3's quoted vLLM caveat) — it asserts only configuration comparability,
  and `hardware` is exactly the field a consumer checks before treating comparable results as
  reproducible.
- **`warnings` — included, but narrowed to its stable identity fields.** Rationale for inclusion is
  unchanged: a degraded run is not comparable to a clean one. But `ResponseWarning`'s own docstring
  (OS-4/DEC-053, pre-existing) already draws this exact line: `code` is "the stable machine
  identifier... what `strict` keys on," while `message` is "human text and the only field free to
  change without a spec change." Hashing `message` would contradict that pre-existing contract — an
  operator rewording a warning string must not silently break comparability for every historical
  occurrence of that degradation. So the preimage carries `type`/`code`/`field` per warning; `message`
  never participates.

This refines §8.3's literal "everything outside `timing`" for C1 in three ways — `batch`'s exclusion,
`hardware`'s exclusion, and `warnings`' narrowed granularity — each argued above rather than merely
inherited from a block list.

### D3 — `X-Run-Id` is non-streaming only in C1
`run_id` cannot be known when SSE headers are flushed (the manifest is finalized post-generation).
DEC-053's metadata lifecycle already forbids emitting not-yet-determined facts in the single
pre-generation event. So C1 emits `X-Run-Id` on **non-streaming** responses only; a streaming
manifest surface (a terminal SSE event) is deferred. Varex uses non-streaming, so C5 is unaffected.

### D4 — Provenance is honest but its completeness is not finalized — [OPEN]
C1 forbids fabrication: any provenance value the platform cannot determine is omitted or `null`,
never approximated. C1 does **not** decide whether specific provenance fields (`weights_sha256`,
`hf_revision`) are mandatory, nor whether their absence emits a `warning`. That policy is **[OPEN]**
and deferred. New provenance C1 *does* commit to (cheap, always available): `engine.git_sha`,
`engine.backend_version`, `request.prompt_sha256`, `request.chat_template_sha256`, and extended
`hardware` (`driver`, `cuda`, `wsl2` via existing `_is_wsl`, `cpu` name) — each under the same
never-fabricate rule.

### D5 — Manifest delivery beyond the header is deferred — [OPEN]
§8.3 says a response "carries, *or references by content hash*," a manifest. C1 commits only to
`X-Run-Id`. Whether the full manifest also rides the response body (additive optional field, per the
B3 `timing` precedent) and/or is retrievable via `GET /v1/manifest/{run_id}` is **[OPEN]** — the
immediate follow-up, not decided here. C1 defines the manifest object and its identity; delivery is
a separable increment.

### D6 — [OPEN] Does `batch` (incl. populated `co_batched_request_ids`) enter the `run_id` preimage?
When C2 populates `co_batched_request_ids`, including it in the preimage would make every concurrent
request produce a distinct `run_id` (co-batch membership varies per execution) — shifting `run_id`'s
meaning from "same recorded *configuration*" to "identity of one specific batched *execution*." That
is a fundamental question about what `run_id` denotes. **Deferred to C2**; C1 excludes `batch`
entirely (D2) so the decision stays open.

### D7 — [OPEN] Residual "effective seed" phrasing in a neighbouring OS-4 requirement
The `platform` requirement *Substitutions are visible in the response* → scenario *Client can
reconstruct what ran* currently reads "…including **the effective seed**." Under this change's
terminology that value is the **requested** seed (echoed unchanged; none when unsupplied). C1
MODIFIES only *Optional request seed reaches the sampler* (the primary target of the approved
amendment) and does **not** edit *Substitutions are visible* — flagging the terminology
reconciliation as **[OPEN]** for the change owner rather than expanding scope unasked.

## Risks / Migration

- **Additive**, so no consumer migration. `X-Run-Id` is ignorable.
- **The seed amendment supersedes stale OS-3 text** (the `MUST NOT echo the effective seed` clause
  and the "No response echo" scenario, both already contradicted in-tree by OS-4's shipped
  `resolved`). This is spec-hygiene catch-up, not a behavior break. Record a DEC on archive.
- **vLLM coupling** for `git_sha`/`backend_version`/`runtime` fields is read-only reflection of
  `vllm_config` and package metadata — no engine-internals patching in C1 (that risk is C2's).
