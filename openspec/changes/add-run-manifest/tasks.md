# Tasks — add-run-manifest (C1)

> Implementation checklist for the approval-gated build phase. Drafting this list is not
> implementation; do not check items until the corresponding code lands.

## 1. Run identity
- [x] 1.1 Add `run_id` computation in `utils/ids.py` (currently empty): `sha256` over the canonical
      preimage; reuse `benchmarks/suite_identity.py` canonicalization discipline (D1).
- [x] 1.2 Define the preimage assembler = `{engine, model, runtime, sampling, request, warnings}`,
      excluding `timing`, `batch`, and `hardware` (D2). `warnings` is projected onto each warning's
      `type`/`code`/`field` — `message` is excluded (D2, amended after implementation review).
- [x] 1.3 Unit tests (mockable, no GPU): identical inputs → identical `run_id`; a non-timing field
      change → different `run_id`; a `timing`-only change → same `run_id`; `run_id` is recomputable
      from the manifest (content hash, not random).

## 2. Manifest object
- [x] 2.1 Add the manifest schema to `schemas/chat.py` (`manifest_version` + blocks), reusing
      `ResponseWarning`, `ResolvedRequest`, `EngineTiming`.
- [x] 2.2 Assemble the manifest in `services/chat_service.py` from existing per-request signals — no
      re-derivation of tokens/timing/warnings.
- [x] 2.3 New provenance: `engine.git_sha`, `engine.backend_version`, `request.prompt_sha256`,
      `request.chat_template_sha256`, extended `hardware` (`driver`/`cuda`/`wsl2`/`cpu`) — each
      omitted or `null` when unavailable, never fabricated (D4).
- [x] 2.4 Reserve `batch.co_batched_request_ids` in the schema; do NOT populate from engine
      internals (C2 deferred).
- [x] 2.5 `sampling.seed` records the requested seed only; `null` when the client supplied none.

## 3. Header
- [x] 3.1 Emit `X-Run-Id` on non-streaming `POST /v1/chat/completions` responses (D3).
- [x] 3.2 Test: header present and equals the manifest's `run_id`; streaming responses carry no
      manifest surface in this change.

## 4. Seed-echo amendment
- [x] 4.1 No behavior change required beyond docs/manifest — the requested-seed echo already exists
      (`resolved.seed`). Ensure `manifest.sampling.seed` follows the same requested-only rule.
- [x] 4.2 Update any docs claiming/near-claiming determinism to assert comparability only.
      No live doc (README/API docs) currently overclaims determinism from seed or the manifest;
      the only current description of this contract is the spec.md delta itself, already worded
      as comparability-only. No edit target found — see final report.

## 5. Validation & docs
- [x] 5.1 `openspec validate add-run-manifest --strict` passes.
- [x] 5.2 Unit suite, `ruff`, `mypy` green (mockable tests only; no GPU path in this change).
- [ ] 5.3 On archive: record a DEC superseding the stale OS-3 seed-echo clause; note the D6/D7
      `[OPEN]`s for the change owner.
