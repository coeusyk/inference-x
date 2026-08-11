# Tasks — integrate-varex-manifest (C5)

## 1. InferenceX — resolve D5 (inline manifest delivery)
- [x] 1.1 Add `manifest: Optional[RunManifest]` field to `ChatCompletionResponse`
      (`schemas/chat.py`), defaulting to `None`, non-streaming only, following the `resolved`/
      `timing` precedent.
- [x] 1.2 Wire it in `ChatService.complete()` (`chat_service.py:282`): include `"manifest":
      manifest` in the same `model_copy(update=...)` call that already sets `run_id`.
- [x] 1.3 Integration test (`tests/unit/test_routes.py`, GPU-free, `TestClient` + stub engine):
      POST non-streaming, assert `resp.json()["manifest"]["run_id"] == resp.json()["run_id"] ==
      resp.headers["X-Run-Id"]`, then recompute `run_id` from the manifest's own
      `engine`/`model`/`runtime`/`sampling`/`request`/`warnings` fields via `compute_run_id`
      (projecting warnings to `type`/`code`/`field` only, matching `_build_manifest`) and assert
      it equals the returned `run_id` — this is the boundary comparability check (design.md D2),
      not a serialization round-trip.
- [x] 1.4 Streaming responses: confirm (existing behavior, no new test needed beyond the existing
      `test_streaming_response_carries_no_x_run_id_header`) that `manifest` is absent/`None` since
      the streaming path never calls `_build_manifest`.

## 2. Varex — client capture
- [ ] 2.1 `OpenAICompatClient.generate()`: after parsing `content`, also set
      `self.last_run_id = data.get("run_id")` and `self.last_manifest = data.get("manifest")`
      (both `None` when absent — every non-InferenceX provider). No `ModelClient` Protocol change
      (design.md D4).
- [ ] 2.2 `__init__` initializes `self.last_run_id: str | None = None` and
      `self.last_manifest: dict | None = None` alongside the existing `last_token_count = 0`.
- [ ] 2.3 Test (`tests/test_models.py`, `pytest-httpx`, no live GPU): a mocked InferenceX-shaped
      response (`run_id` + `manifest` present) is captured; a mocked OpenAI-shaped response
      (neither present) leaves both `None` — degrade, never fabricate.

## 3. Varex — persistence
- [ ] 3.1 `cli.py::_run_loop`: capture `run_id_a = getattr(gen_client, "last_run_id", None)`
      immediately after the `answer_a` generate call, `run_id_b` immediately after `answer_b` —
      same position as the existing `total_tokens += getattr(gen_client, "last_token_count", 0)`
      lines, and safe for the same reason (sequential, not gathered).
- [ ] 3.2 Pass `run_id_a`/`run_id_b` through to `store.write_trial(...)`.
- [ ] 3.3 `storage/store.py`: add nullable `run_id_a: str`, `run_id_b: str` to the `trials.create()`
      column dict for brand-new databases, AND add the `_ensure_schema()` migration for
      already-existing `trials` tables (design.md Risks) so `--resume` against a pre-change
      experiment DB doesn't break.
- [ ] 3.4 `write_trial()` gains `run_id_a: str | None = None, run_id_b: str | None = None` keyword
      params, inserted into the new columns.
- [ ] 3.5 Test (`tests/test_storage.py`): `write_trial()` with run_ids persists and reads back
      correctly; a migration test opens a `Store` against a hand-built DB whose `trials` table
      lacks the new columns and confirms `_ensure_schema()` adds them without data loss.

## 4. Varex — provider registry + docs
- [ ] 4.1 `models/registry.py`: add an `inference-x` entry to `PROVIDER_PRESETS`
      (`protocol="openai_compat"`, `base_url` pointing at a local InferenceX server), per the
      registry's own documented one-entry addition pattern.
- [ ] 4.2 `benchmarks/README.md`: bump the documented `Manifest fields` section to
      `schema_version: 2`, adding optional `run_id`/sample-manifest sub-fields under the existing
      `generator`/`judge` blocks. Note that `manifest.yaml` remains hand/agent-assembled — this is
      a documentation-only convention change (design.md D5), not a new archival tool.
- [ ] 4.3 `TRACKER.md`: add an entry recording this cross-repo decision (schema_version bump,
      `run_id_a`/`run_id_b` columns, `inference-x` preset) per Varex's own "TRACKER.md is the
      living document" convention.

## 5. Validation
- [ ] 5.1 InferenceX: `openspec validate integrate-varex-manifest --strict`, full pytest, ruff,
      mypy — all green.
- [ ] 5.2 Varex: full `pytest` (including the two new test additions) green.
- [ ] 5.3 **Not covered by this change** — the live SPRT experiment against a GPU-backed
      InferenceX server (proposal.md "Explicitly out of scope"). Owner's remaining step before
      C5's stated exit evidence is met.
