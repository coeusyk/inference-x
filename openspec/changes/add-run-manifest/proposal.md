# Proposal: add-run-manifest (Phase C — C1)

## Why

Phase C's thesis (`docs/REVIEW-2026-08-03-architecture.md` §8.1, §11) is that a third party, given
only the output artifact, can (a) tell exactly what computation produced a result, (b) decide
whether two results are comparable, and (c) reproduce it or know precisely why not. No surveyed
engine does this. The differentiating artifact is the **signed run manifest** of §8.3 — with
`run_id` as a content hash that makes results *addressable* and comparability a mechanical check.

Phase A/B already produced most of the manifest's inputs: typed degradation (`warnings`, DEC-052),
the effective request echo (`resolved`, DEC-053), per-request engine timing (`EngineTiming`, B3),
forwarded `seed` (DEC-051), and a hardware profiler (`benchmarks/hardware.py`). What does not exist:
a **content-addressed run identity** (`run_id` + `X-Run-Id`), a **manifest object** that assembles
those signals into one attestable record, and the honest provenance/comparability contract around
them. `utils/ids.py` is empty; there is no manifest anywhere in `src/`.

This change (C1) establishes that identity and object. It is the keystone the rest of Phase C writes
into (C2 batch composition, C3 `runtime.batch_invariant`, C5 Varex embedding).

## What Changes

- **Run identity.** Define `run_id = "sha256:" + hash(canonical manifest preimage)`. The preimage is
  the semantics-determining manifest fields; it **excludes** the speed/observational blocks. Two
  runs are comparable iff their `run_id`s match. **Content-addressing only** — no cryptographic
  signature is produced in this change (the word "signed" in the roadmap is aspirational; signing is
  a Phase E candidate).
- **`X-Run-Id` header** on **non-streaming** `POST /v1/chat/completions` responses.
- **Run manifest object.** A `manifest_version`-stamped record assembling existing per-request
  signals (`resolved`, `warnings`, `timing`, requested `seed`, hardware) plus new provenance
  derivations (`engine.git_sha`, `engine.backend_version`, `request.prompt_sha256`,
  `request.chat_template_sha256`, extended `hardware`). Provenance is honest: a value the platform
  cannot determine is omitted or `null`, never fabricated.
- **Reserved batch field.** `batch.co_batched_request_ids` is defined in the schema but **not
  populated** from engine internals in this change (C2, deferred until C5 establishes value).
- **Seed-echo amendment.** Reconcile the stale OS-3 prohibition with the OS-4 `resolved` echo:
  the platform MAY echo the **requested** seed as provenance (resolved block and manifest), MUST NOT
  echo or fabricate a **backend-derived effective** seed, and MUST NOT claim reproducibility from the
  seed alone.
- **No overclaim.** The manifest and its documentation assert the §8.3 *comparability* property, not
  reproducibility. The scoped determinism contract is deferred to C3.

## Impact

- **Additive & backward-compatible.** A new response header and a new manifest object; existing
  response fields and semantics are unchanged when the manifest is unused. Consumers that ignore
  `X-Run-Id` are unaffected.
- **Spec:** MODIFIES `platform` requirement *Optional request seed reaches the sampler*; ADDS run
  identity, manifest content, and no-overclaim requirements to `platform`.
- **Affected code (implementation phase, not this proposal):** `utils/ids.py` (new `run_id`),
  `schemas/chat.py` (manifest schema + `X-Run-Id` wiring point), `services/chat_service.py`
  (assemble from `resolved`/`warnings`/`timing`), `engines/vllm_engine.py` (git_sha /
  backend_version / runtime fields), `benchmarks/hardware.py` (extended hardware), reuse
  `benchmarks/suite_identity.py` canonicalization discipline for hashing.
- **Explicitly out of scope (deferred):** manifest *delivery* beyond the header (inline body field
  and/or `GET /v1/manifest`), `weights_sha256`/`hf_revision` completeness policy, ITL timing,
  determinism (C3), plan/doctor (C6), Varex (C5), populating `co_batched_request_ids` (C2), and any
  streaming manifest surface.
