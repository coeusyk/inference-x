# Design — integrate-varex-manifest (C5)

## Context

Authoritative sources: `docs/REVIEW-2026-08-03-architecture.md` §8.3 (manifest JSON, Varex
evaluated as the downstream consumer), the Phase C planning report §4.6 ("Varex run evidence —
C5") and §9 (sequencing: C5 before C2), `openspec/changes/archive/*/add-run-manifest/design.md`
D5 (the open question this change resolves), and Varex's own `ARCHITECTURE.md` / `TRACKER.md`
conventions. This design covers both repos because the change is intentionally cross-repo — there
is no InferenceX-only or Varex-only version of "Varex can consume a run manifest."

## Goals / Non-Goals

**Goals**
- Resolve D5: decide how the full manifest reaches a client, and implement it.
- Give Varex a way to capture and persist `run_id` per generation call without changing its
  `ModelClient` Protocol or its evaluation semantics.
- Make `run_id` independently verifiable (recomputable) at the boundary, not just trusted.
- GPU-free integration tests on both sides that exercise the real call path, not isolated
  serialization.

**Non-Goals**
- C2 (populating `batch.co_batched_request_ids`) — stays evidence-gated behind this change.
- A `GET /v1/manifest/{run_id}` retrieval endpoint (see D1).
- A `deterministic` field in Varex's `ProviderConfig` (see D3).
- Any Varex archival automation (`--archive` command, auto-writing `manifest.yaml`).
- Running the live SPRT experiment (see D5 below, this document's own D5 — Varex's TRACKER
  numbering is independent of C1's; not to be confused with C1 design.md's D5, which this document
  resolves).

## Decisions

### D1 — Resolve C1's D5 via inline body field, not a retrieval endpoint

C1's D5 named two candidates: the manifest additionally rides the response body, or it is
retrievable via `GET /v1/manifest/{run_id}`. The retrieval endpoint requires new state — a
manifest store, with a retention/eviction policy that D5 never specced and that would itself be a
genuine new owner decision (how long to retain, in-memory vs. persisted, eviction under what
pressure). The inline field requires none of that: the manifest object is already fully assembled
in `ChatService.complete()` before the response is returned (`chat_service.py:273`); attaching it
costs one field and one `model_copy(update=...)` key. It also matches the precedent already set
twice in this response model — `timing` (B3) and `resolved` (OS-4) are both "data the service
already has, exposed as an additive optional field." Inline wins on every axis that matters here:
less new surface, direct precedent, and it is what C5 actually needs (Varex reads response bodies
today; it has no header-reading hook and does not need one added for this).

### D2 — `run_id` becomes mechanically verifiable, not just asserted

Because the manifest rides the body, a client holding both `run_id` and the manifest can recompute
`run_id` from the manifest's own `engine`/`model`/`runtime`/`sampling`/`request`/`warnings` fields
(same preimage projection `chat_service._build_manifest` uses — warnings reduced to
`type`/`code`/`field`, `message` excluded) and confirm it matches. This is the C1 comparability
claim (§8.3: equal `run_id` ⇒ same recorded configuration) made checkable at the boundary instead
of trust-me. The InferenceX-side integration test (tasks.md §2) does exactly this via
`TestClient` + `compute_run_id`, without needing a GPU.

### D3 — Varex does not gain a `deterministic` field

The planning report's C5 exit evidence is "one real SPRT experiment end-to-end with a **pinned
seed**" — not a claim that the run is deterministic. Varex's `ProviderConfig` already has an
optional `seed`. Making InferenceX's response *comparable* run-to-run under a pinned seed requires
the InferenceX process to have been booted with `INFERENCE_X_DETERMINISTIC=1` (Phase C3) — that is
an operational prerequisite for whoever runs the live experiment, documented in this change's
tasks, not a Varex schema or code change. Adding a `deterministic` field to `ProviderConfig` would
imply Varex can toggle it per-request, which it cannot (C3 requires process-startup activation) —
that would be a misleading contract, so it is deliberately not added.

### D4 — Varex client capture uses an untyped attribute, not a Protocol change

`ModelClient` (`models/base.py`) is a `Protocol` whose `generate()` returns `str`. Every existing
caller (`cli.py::_run_loop`, `evaluation/judge.py::_call_judge`) already reads auxiliary
per-call data (`last_token_count`) via `getattr(client, "last_token_count", 0)` — the Protocol
itself does not declare that attribute; it is a convention, not an interface member. `last_run_id`
and `last_manifest` follow the identical convention: set unconditionally by `OpenAICompatClient`
(to `None` when the response carries no `run_id`/`manifest` — e.g., every non-InferenceX
provider), left unset on `OllamaClient` (callers already default via `getattr(..., None)`, so
omission degrades safely), read via `getattr(gen_client, "last_run_id", None)` in `_run_loop`
immediately after each `await generate()` call.

This is safe specifically because `_run_loop` awaits the prompt-A and prompt-B generate calls
**sequentially on one client instance** (`answer_a = await gen_client.generate(...)`; only then
`answer_b = await gen_client.generate(...)` — `cli.py`, no `asyncio.gather`), so there is no window
where a second call's `last_run_id` could clobber the first before it is read — exactly the
existing race-freedom the `last_token_count` billing code next to it already depends on. Had the
two calls been gathered concurrently, this pattern would be unsafe and `generate()`'s return type
would need to change instead; that is not the case here.

### D5 — `manifest.yaml`'s schema_version bump is documentation, not code

Confirmed by exhaustive grep (`grep -rln -i manifest src tests docs configs` → no hits) that no
code under Varex's `src/` writes `manifest.yaml`; `benchmarks/README.md`'s "How to archive a run"
section describes a manual/agent-driven procedure (freeze `report.json`, write `manifest.yaml` by
hand, append to `INDEX.yaml`). So "schema bump 1→2" cannot mean changing a serializer — there is
none. It means: (a) document the new optional `run_id`/sample-manifest sub-fields under the
existing `generator`/`judge` blocks in `benchmarks/README.md`, and (b) make sure the data those
sub-fields would reference is actually captured and persisted during the run (the new
`trials.run_id_a`/`run_id_b` SQLite columns), so a future archiver has something real to transcribe
instead of reconstructing it after the fact (impossible — `run_id` is a hash of ephemeral
request/response state, not derivable post hoc). Building an `--archive` command that writes
`manifest.yaml` automatically is out of scope (proposal.md).

## Risks / Migration

- **InferenceX:** additive only. `manifest: Optional[RunManifest] = None` defaults to `None` on
  direct construction (matching `resolved`'s existing precedent), so no caller outside
  `ChatService.complete()` needs updating. Non-streaming JSON response bodies grow by the size of
  one `RunManifest` — acceptable; no client is forced to parse it.
  Streaming responses gain no new field.
- **Varex:** `write_trial()` gains two new optional keyword parameters with defaults, so existing
  callers keep working. The SQLite schema addition needs an explicit migration, not just a
  `CREATE TABLE IF NOT EXISTS`: `_ensure_schema()`'s existing guard (`if "trials" not in
  table_names()`) only runs for a brand-new database file — a `--resume` against an experiment DB
  created before this change would open a `trials` table that already exists (skipping creation)
  and then fail on the first `write_trial()` insert carrying `run_id_a`/`run_id_b` keys the table
  doesn't have. `_ensure_schema()` therefore also checks, when `trials` already exists, whether
  those two columns are present and adds them (nullable, `sqlite_utils.Table.add_column`) if not —
  an idempotent migration, not a documentation note.
- **No signing, no new stateful services, no engine-boundary changes** on either side.
