# Proposal: integrate-varex-manifest (Phase C — C5)

## Why

Phase C's exit evidence for the manifest keystone (C1, `add-run-manifest`) is a downstream
consumer that actually uses it. Varex (`~/projects/varex`, SPRT prompt evaluation) is that
consumer (`docs/REVIEW-2026-08-03-architecture.md` §8.3, planning report §4.6). Varex's own
`manifest.yaml` archival format (schema_version 1) records only provider/model *labels*
(`generator`/`judge` blocks) — no run identity, no seed, no hardware, no way to tell whether two
archived experiments ran against comparable configurations. That gap is exactly what C1's `run_id`
closes, but nothing lets Varex reach it today.

Investigation of this repo surfaced the actual blocker: C1's own `design.md` left **D5 open**
("manifest delivery beyond the header is deferred") — `run_id` reaches a client via `X-Run-Id`,
but the full manifest object is never serialized into the response body, and no
`GET /v1/manifest/{run_id}` exists. Varex's `OpenAICompatClient` is a generic HTTP client with no
header-reading hook. Without resolving D5, C5 cannot be specced, because there is nothing for
Varex to consume beyond an opaque string.

## What Changes

- **Resolves C1's [OPEN] D5** by choosing the inline-body option D5 named (not the
  `GET /v1/manifest/{run_id}` alternative — that would require a manifest store with a retention
  policy, which is a genuine new piece of state C5 does not need and D5 never specced). This is
  additive: a new optional `manifest` field on `ChatCompletionResponse`, non-streaming only, using
  the exact precedent already set by `timing` (Phase B3) and `resolved` (OS-4) — existing fields
  the response already carries, exposed as-is, no new derivation.
- **Makes `run_id` mechanically verifiable at the InferenceX→Varex boundary**: with the manifest
  inline, any client can recompute `run_id` from the returned preimage and confirm it matches the
  returned `run_id`, rather than trusting the string. This is the C1 comparability claim made
  checkable rather than asserted.
- **Varex-side (this repo does not own Varex's code; changes there are implemented in the same
  pass per the user's cross-repo instruction, recorded in Varex's own `TRACKER.md` since Varex has
  no OpenSpec):**
  - `OpenAICompatClient.generate()` captures `run_id`/`manifest` from the response body into
    `self.last_run_id` / `self.last_manifest`, following the exact pattern already used for
    `self.last_token_count` (an untyped instance attribute read via `getattr(..., None)` by
    callers — `ModelClient`'s `Protocol` does not declare `last_token_count` either, so this adds
    no interface surface). Absent on non-InferenceX providers — degrades to `None`, never
    fabricated.
  - `cli.py`'s `_run_loop` captures both prompt-A and prompt-B `run_id`s immediately after each
    `await gen_client.generate(...)` call — the two calls are sequential, not `asyncio.gather`ed,
    so this is race-free, exactly like the existing `last_token_count` billing code it sits next
    to.
  - `storage/store.py`'s `trials` table gains two nullable columns (`run_id_a`, `run_id_b`) so a
    persisted experiment can later be checked for configuration comparability across trials.
  - `registry.py` gains an `inference-x` entry in `PROVIDER_PRESETS` (documented as a one-entry,
    no-other-files-change addition already).
  - `benchmarks/README.md`'s manifest schema doc bumps to `schema_version: 2`, documenting optional
    `run_id`/sample-manifest sub-fields under the existing `generator`/`judge` blocks. `manifest.yaml`
    is hand/agent-assembled at archive time, not code-generated (confirmed: no code under `src/`
    writes it) — the bump is a documented convention change plus ensuring the underlying data is
    now actually captured and persisted (via the two new columns), not a new archival tool.

## Impact

- **Additive & backward-compatible** on both repos. No existing field, header, or CLI flag changes
  meaning. Consumers ignoring `manifest`/`last_run_id` are unaffected.
- **Spec:** ADDS one requirement to InferenceX's `platform` capability (manifest rides the
  non-streaming response body). No MODIFIED requirements — the existing `X-Run-Id` header
  requirement is unchanged; this is a second, independent delivery surface for the same object.
- **Affected code:**
  - InferenceX: `schemas/chat.py` (`ChatCompletionResponse.manifest`), `services/chat_service.py`
    (`complete()` return), `tests/unit/test_routes.py` (integration test).
  - Varex: `src/models/openai_compat.py`, `src/cli.py` (`_run_loop`), `src/storage/store.py`,
    `src/models/registry.py`, `benchmarks/README.md`, `TRACKER.md`, `tests/test_models.py`,
    `tests/test_storage.py`.
- **Explicitly out of scope:**
  - **C2** (co-batch extraction / populating `batch.co_batched_request_ids`) — remains
    evidence-gated behind this change's results, per DEC and the planning report; not touched.
  - A `GET /v1/manifest/{run_id}` retrieval endpoint — the other D5 option; not needed once the
    body carries the manifest inline, and it would add state (a manifest store + retention policy)
    this change does not need.
  - Plumbing a `deterministic` field into Varex's `ProviderConfig` — the planning report's C5 exit
    evidence names a **pinned seed**, not `deterministic: true`; Varex already sends `seed`.
    Determinism requires InferenceX to boot with `INFERENCE_X_DETERMINISTIC=1` (Phase C3), which is
    an operational note for whoever runs the live experiment, not a schema change.
  - A Varex `--archive` command or any other automation of `manifest.yaml` production — archival
    stays the existing hand/agent-assembled process; this change only ensures real data
    (`run_id`) is available to put into it.
  - **The live SPRT experiment itself.** This change makes the contract and plumbing real and
    verifies it with GPU-free integration tests on both sides (InferenceX: `TestClient` + a stub
    engine; Varex: `pytest-httpx` mocking an InferenceX-shaped response). It does not run a live
    ~200-trial experiment against a GPU-backed InferenceX server — that is the owner's remaining
    step before C5's stated exit evidence is met, and it is also what would validate DEC-061's
    batch defaults at scale and unblock C2.
