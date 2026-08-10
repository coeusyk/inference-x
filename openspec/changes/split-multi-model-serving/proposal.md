# Proposal: split-multi-model-serving

## Status

**Investigation complete, including a live VRAM/GPU experiment on real dev
hardware. Recommendation upgraded from conditional to supported, not yet
decided.** The prior round's recommendation (Option A, design.md §10) was
conditional on an untested VRAM-cost question. This round ran that
experiment on the actual dev box (RTX 4060, 8188 MiB, `6gb` tier) against
real cached models. The narrow original question (CUDA-graph VRAM cost)
could not be cleanly measured — both the status-quo pool path and the
two-process alternative failed identically for the tight DEC-046 reference
pair, from a shared, pre-existing free-VRAM-probe bug unrelated to either
option (design.md §6). But Option A's actual mechanism was independently,
positively demonstrated: two independent single-model processes, graphs on,
loaded successfully with real VRAM headroom for a real (lighter) pair
(design.md §6 finding 5). Nothing found this round argues against Option A
relative to the status quo — the one bug found affects both equally. The
recommendation is now supported, with two concrete, scoped implementation
prerequisites rather than open architectural questions (design.md §10).

**Both change-owner decisions approved (`tasks.md` §2): Option A, with both
prerequisites in scope. Implementation is complete** (`tasks.md` §3-6):
the free-VRAM probe now queries `nvidia-smi`; the multi-engine sequential
VRAM heuristics, `enforce_eager` coupling, and the `max_model_len` 2048
clamp are deleted; `_build_engine_pool` rejects more than one distinct
model instead of silently loading a second in-process engine; the
playground TUI orchestrates one process per compared model. Full record:
`docs/DECISIONS.md` DEC-059. Validation: 550 unit tests pass, `ruff`/`mypy`
clean, live GPU re-verification (design.md §12) confirms the probe fix and
reproduces the positive two-process result end-to-end under the shipped
code. Two follow-up experiments remain genuinely open and non-blocking
(crash isolation, design.md §11 item 2; probe-bug root cause, item 6). This
change is merge-ready but not yet archived (`tasks.md` §6.4) — archived
only after B4/B5 archive, per the layering convention its spec delta
follows.

## Why

Phase-B roadmap item **B6**, as named in `docs/PHASE-A-EXECUTION-PLAN.md:273`
("**Multi-model process split**") and the architecture review's roadmap
table (`docs/REVIEW-2026-08-03-architecture.md:504`): **"Split multi-model
into two processes; delete the global step lock, `enforce_eager` coupling,
and the 2048 pool clamp."** §3.5 of that review names this "a demo feature
charging architectural rent": `EnginePool`'s `pool_size > 1` path exists
solely to let the TUI's side-by-side "compare" mode load two models into
one process, but every model sharing that process pays three costs for the
whole session, not just while actively compared — see design.md §2 for the
full, source-verified account.

**One correction to the roadmap's own wording, established this round:**
the "delete the global step lock" clause is now moot. `_POOL_STEP_LOCK` was
already deleted outright by B1 (`migrate-async-llm-engine`, DEC-058, merged
2026-08-05) — `AsyncLLM` gives each engine its own background engine-core
process, so the in-process step-serialization lock the old `EngineDriver`
needed has no remaining equivalent (**[verified]**, `grep` of
`engines/vllm_engine.py` finds no lock of any kind touching engine
stepping). B6's real remaining scope is narrower than the original roadmap
sentence: `enforce_eager` coupling, the `max_model_len` clamp, the VRAM
sequential-loading heuristics (`utils/vllm_pool_config.py`), and one
B2-introduced Prometheus label-collision risk not yet named in the original
roadmap wording (design.md §2).

B4 and B5 (both merged to `develop`, both explicitly investigated their own
boundary against this phase) independently confirmed zero dependency in
either direction between admission control and B6's process-split work —
this investigation confirms that finding still holds and treats it as an
established invariant, not something to re-derive (design.md §8).

## What Changes (implemented)

- Retired the `pool_size > 1` single-process path: `enforce_eager` coupling,
  the `max_model_len` 2048 clamp, and `utils/vllm_pool_config.py`'s
  multi-engine sequential-VRAM heuristics (`_weight_scaled_utilization`,
  `_apply_sequential_vram_caps`, `_multi_engine_overhead_gib`) are deleted;
  `validate_pool_fits`/`scale_model_config_for_pool` are renamed to
  `validate_model_fits`/`scale_model_config` (single-model only).
- `api/deps.py`'s `_build_engine_pool` constructs exactly one engine per
  process and rejects (`ValueError`) when `INFERENCE_X_LOADED_MODELS`
  resolves to more than one distinct model, instead of silently loading a
  second in-process engine.
- The free-VRAM probe (`probe_gpu_memory_gib`) now queries `nvidia-smi`
  first, falling back to `torch.cuda.mem_get_info()` with a loud warning
  only when `nvidia-smi` is unavailable — closing the staleness bug design.md
  §6 documented.
- Multi-model serving (the TUI's compare mode) is achieved by running one
  InferenceX server process per model, each on its own port —
  `playground/client.py`'s existing `--base-url-a`/`--base-url-b`
  dual-process compare mode already proved this pattern works
  (**[verified]**, design.md §4) and required zero changes.
  `playground/server_control.py`'s `ensure_models_loaded()` now orchestrates
  one process per requested model automatically, so `make playground` stays
  a one-command experience.

See design.md §10 "Recommended direction," §11 "Remaining items," and §12
"Post-implementation verification" for the full evidence and decision
record, and `docs/DECISIONS.md` DEC-059 for the canonical decision entry.

## Impact

- Affected code: `src/inference_x/utils/vllm_pool_config.py`,
  `src/inference_x/engines/vllm_engine.py`,
  `src/inference_x/utils/cuda_env.py`, `src/inference_x/api/deps.py`,
  `src/inference_x/services/chat_service.py`, `playground/server_control.py`,
  `playground/app.py`. `src/inference_x/engines/pool.py` is unchanged — it
  was already pool-size-agnostic.
- Affected spec: `openspec/specs/platform/spec.md`, requirement
  "Incremental architecture" — see this change's `specs/platform/spec.md`
  delta (written against B4's and B5's own target end-state text, per the
  layering convention both of those changes already established).
- Explicitly NOT affected: `routing/admission.py`, `core/settings.py`'s
  admission settings, `services/chat_service.py`, B4's semaphore/waiter
  mechanism, B5's batch-priority mechanism — see design.md §8.
- Explicitly out of scope for this change (design.md §9): Phase C (signed
  run manifests, `X-Run-Id`, `co_batched_request_ids`,
  `deterministic`/`VLLM_BATCH_INVARIANT`, the oracle/conformance suite, the
  Varex end-to-end run, `make plan`/`make doctor`); Phase D (a second
  inference backend, an engine factory, optional-`vllm`) — DEC-047
  explicitly reserves that authorization to a future ADR this change does
  not provide; the model registry, variant routing, and benchmark advisor
  subsystems.
