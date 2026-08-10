# Tasks: split-multi-model-serving

**Section 1 (Investigation) is complete and closed.** No further
investigation is planned or required to reach a recommendation — design.md
§10's Option A recommendation is supported, not conditional (§6's live
experiment). Implementation is blocked on exactly two change-owner
decisions in Section 2 below, nothing else. 1.15 and 1.18 are open
follow-up experiments, not investigation blockers — see §2.3. Sections 3-6
are scaffolded below for structural continuity with B4/B5's own tasks.md
convention but contain no completed work. Nothing has been implemented, no
branch created, nothing committed.

## 1. Investigation

- [x] 1.1 Read `engines/pool.py` in full; document `EnginePool`'s exact
      construction, dispatch, and lifecycle contract (design.md §1).
- [x] 1.2 Read `api/deps.py`'s `_build_engine_pool` and
      `_build_admission_controller` in full; confirm how `pool_size` and
      per-model config are resolved and threaded into `VLLMEngine`
      construction (design.md §1).
- [x] 1.3 Read `engines/vllm_engine.py`'s construction path in full; confirm
      the exact `pool_size`-conditional kwargs (`enforce_eager`, the
      Prometheus-collision warning) and grep for any remaining step-lock
      code (design.md §2).
- [x] 1.4 Read `utils/vllm_pool_config.py` in full (558 LOC); document every
      function gated on `pool_size > 1` / `len(model_configs) > 1`
      (`_weight_scaled_utilization`, `_apply_sequential_vram_caps`,
      `_multi_engine_overhead_gib`, `validate_pool_fits`,
      `scale_model_config_for_pool`'s 2048 clamp) (design.md §2).
- [x] 1.5 Read `engines/base.py` (`BaseEngine`) in full; confirm the Engine
      Boundary contract has zero `pool_size`/multi-engine surface
      (design.md §7).
- [x] 1.6 Read `docs/REVIEW-2026-08-03-architecture.md` §3.5 and the roadmap
      table (line 504) as the canonical B6 problem statement; read
      `docs/PHASE-A-EXECUTION-PLAN.md`, `docs/PHASE-A-ARCHITECTURE.md`, and
      `docs/REVIEW-2026-08-04-phase-a-final-audit.md` for every other B6
      cross-reference (design.md, throughout).
- [x] 1.7 Read `docs/DECISIONS.md` DEC-058 (B1/`AsyncLLM` migration) in
      full; confirm the step-lock deletion and its explicit "B6 remains the
      only phase authorized to redesign multi-engine serving" framing
      (design.md §1, §2).
- [x] 1.8 Read `docs/DECISIONS.md` DEC-045/046 (VRAM estimation heuristics,
      per-architecture overhead table, the sequential-cap
      double-reservation bug and its fix) as the origin and calibration
      history of the machinery this change would delete (design.md §2).
- [x] 1.9 Read `docs/DECISIONS.md` DEC-047 ("Engine Boundary and backend
      plurality") in full; confirm this change's boundary against Phase D
      backend-plurality authorization (design.md §7, §9).
- [x] 1.10 Read `rescope-admission-control/design.md`'s "Semaphore ownership
      and keying" section and `add-batch-priority-queueing/design.md` §9
      "B6 boundary" in full; confirm both B4 and B5 already independently
      found zero dependency with this change's scope, in either direction
      (design.md §8).
- [x] 1.11 Read `playground/client.py`, `playground/app.py`,
      `playground/startup_screen.py`, and `playground/console.py`; confirm
      the TUI vs. batch-client split on dual-base-url compare support, and
      confirm neither imports server-side code (design.md §4).
- [x] 1.12 Read `CHANGELOG.md`'s B1 entry for the Prometheus
      metrics-registry-collision warning's exact wording, and
      `expose-native-engine-metrics/design.md`'s Non-Goals section
      confirming it is explicitly deferred to this change (design.md §2).
- [x] 1.13 Confirm current `openspec/specs/platform/spec.md`'s "Incremental
      architecture" requirement as the capability this change's spec delta
      must extend, and confirm B4's and B5's own not-yet-archived target-text
      layering convention to build this change's delta against (spec.md
      delta header comment).
- [x] 1.14 **Experiment run this round**, on the actual dev GPU (RTX 4060,
      8188 MiB, `6gb` tier), against real cached models, via `uv run
      uvicorn` server processes measured with `nvidia-smi` (config-only, no
      `src`/`tests`/`pyproject.toml` change). Result: **inconclusive on the
      original narrow question, but positive on the broader mechanism.**
      The tight DEC-046 reference pair (`qwen2.5-0.5b` + `qwen2.5-1.5b`)
      failed identically under both today's pool path AND the two-process
      alternative, from a shared, pre-existing `probe_gpu_memory_gib()`
      staleness bug (design.md §6 findings 1-4) — not a CUDA-graph-cost
      effect. A lighter real pair (`qwen2.5-0.5b` + `tinyllama-chat`) then
      succeeded as two independent processes with real VRAM headroom
      (6543 MiB used / 1414 MiB free of 8188 MiB) — direct positive
      evidence for Option A's mechanism (design.md §6 finding 5, §11 item
      1 revised).
- [ ] 1.15 **Experiment, not run this round:** attempt to reproduce (or find
      existing evidence for) whether a crash inside one co-located
      `AsyncLLM` engine-core process today already stays isolated from its
      sibling engines and the parent FastAPI process, without any B6 change
      (design.md §5, §11 item 2). Weak indirect signal only: a startup
      failure in one process did not destabilize a sibling in this round's
      experiment, but that is not the same claim as mid-flight crash
      isolation.
- [x] 1.16 **Resolved into a concrete implementation requirement this
      round** (no longer an open follow-up): the replacement for
      `validate_pool_fits`'s aggregate pre-flight check must not reuse
      `probe_gpu_memory_gib()` unmodified — it inherits the exact staleness
      bug found in 1.14. Needs an `nvidia-smi`-based or otherwise
      cross-process-aware probe, or explicit non-`"auto"` per-process
      `gpu_memory_utilization` values (design.md §6, §10, §11 item 3).
- [x] 1.17 **Resolved this round:** `GET /v1/models` was verified (source
      read of `api/routes/models.py`, confirmed live in this round's
      experiment) to already report the full static model registry
      regardless of `pool_size`, not the pool's loaded-model list — no
      compatibility change exists here. Only `GET /health`'s `loaded_models`
      field actually changes under one-model-per-process, and that is
      expected, accurate behavior (design.md §7, §11 item 4).
- [ ] 1.18 **New, follow-up only, not blocking:** root cause of
      `torch.cuda.mem_get_info()`'s inaccuracy across sibling
      processes/sequential loads on this platform is not diagnosed
      (reproduced three times, cause unconfirmed — WSL2 accounting, a
      CUDA/driver caching behavior, and a vLLM-version change since
      DEC-046 are all plausible) (design.md §11 item 6).

## 2. Decision — the gate. **Both decisions approved by the change owner.**

- [x] 2.1 **Decision 1 — approve Option A.** Approved: one model per OS
      process, no in-process `EnginePool` multi-engine path.
- [x] 2.2 **Decision 2 — scope the two prerequisites.** Approved: the
      free-VRAM-probe fix and the `validate_pool_fits` replacement ship as
      part of *this* change's implementation (not a separate change).

Non-blocking, informational only — do not gate 2.1/2.2 on these: 1.15
(crash isolation) and 1.18 (probe-bug root cause) remain open follow-up
experiments (design.md §11 items 2 and 6). They can run before, during, or
after implementation without changing the Section 2 decisions above. 1.16
and 1.17 are already resolved and require no further decision.

## 3. Implementation

- [x] 3.1 `utils/vllm_pool_config.py`: `probe_gpu_memory_gib()` rewritten to
      query `nvidia-smi` first, falling back to `torch.cuda.mem_get_info()`
      with a loud warning only when `nvidia-smi` is unavailable; result
      cached for 2s to protect repeat callers (e.g. `GET /v1/metrics`) from
      a subprocess spawn per call.
- [x] 3.2 `utils/vllm_pool_config.py`: deleted `_weight_scaled_utilization`,
      `_apply_sequential_vram_caps`, `_multi_engine_overhead_gib` /
      `_MULTI_ENGINE_OVERHEAD_GIB`.
- [x] 3.3 `utils/vllm_pool_config.py`: `validate_pool_fits` →
      `validate_model_fits` (single model; delegates to
      `_single_engine_utilization` so the preflight and the actual sizing
      share one basis). `scale_model_config_for_pool` → `scale_model_config`
      (drops `pool_size`/`pool_models`/`pool_configs`/`engine_index`/
      `session_free_vram_gib` and the `max_model_len` 2048 clamp).
- [x] 3.4 `engines/vllm_engine.py`: `VLLMEngine.__init__` drops the same pool
      params; deletes `enforce_eager` forcing and the Prometheus
      metric-collision warning; `_probe_cuda_vram()` delegates to the fixed
      `probe_gpu_memory_gib()` instead of duplicating the read.
- [x] 3.5 `utils/cuda_env.py`: `ensure_vllm_runtime_env()` drops the
      `pool_size` param and its `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`
      branch.
- [x] 3.6 `api/deps.py`: `_build_engine_pool` rejects (`ValueError`) when
      `INFERENCE_X_LOADED_MODELS` resolves to more than one distinct model;
      constructs exactly one `VLLMEngine` otherwise.
- [x] 3.7 `services/chat_service.py`: `_resolve_engine`'s not-in-pool error
      message drops the now-inapplicable "add it to the existing list"
      clause.
- [x] 3.8 `playground/server_control.py`: `start_playground_server` takes a
      single `model` + `log_path`; `ensure_models_loaded` orchestrates one
      process per requested model on consecutive ports (index 0 keeps the
      caller's own base URL/port), returns `dict[model, base_url] | None`
      instead of `bool`. `stop_playground_server()`'s blanket kill is called
      at most once per call, before starting any replacement processes.
- [x] 3.9 `playground/app.py`: `self.base_urls` tracks each compared model's
      own base URL; `_refresh_server_state` checks each model's own
      `/health` instead of a shared `/v1/models` registry check;
      `_stream_model` routes to the model's own base URL.
      `playground/chat.py` and `playground/client.py` required zero changes
      (confirmed: both already compatible with the new truthy/falsy and
      dual-base-url contracts respectively).

## 4. Tests

- [x] 4.1 `tests/unit/test_vllm_pool_config.py`: removed the 7
      multi-engine-specific tests; renamed remaining single-model tests off
      `pool_size=1`; added `validate_model_fits` tests (rejects on low free
      VRAM, allows with ample free VRAM, no-op when free VRAM unknown) and
      probe tests (`nvidia-smi` path, torch fallback with warning, no-GPU
      `(None, None)`, 2s cache).
- [x] 4.2 `tests/unit/test_vllm_engine_knobs.py` /
      `tests/unit/test_vllm_max_model_len.py`: patch targets renamed to
      `scale_model_config`; removed the two `pool_size` collision-warning
      tests (the behavior they tested no longer exists).
- [x] 4.3 `tests/unit/test_deps_tier_knobs.py` /
      `tests/unit/test_deps_variant_routing.py`: patch targets renamed to
      `validate_model_fits`; added a new test asserting `_build_engine_pool`
      raises when `INFERENCE_X_LOADED_MODELS` resolves to >1 distinct model.
- [x] 4.4 `tests/unit/test_server_control.py`: rewrote both
      `ensure_models_loaded` tests for the new `dict | None` contract; added
      a new test asserting N-model input starts N processes on N
      consecutive ports.
- [x] 4.5 `tests/unit/test_app.py`: the ctrl+c regression test's fake
      `ensure_models_loaded` now returns `None` (not `False`) on failure,
      matching the new contract.
- [x] 4.6 Live GPU verification (not a unit test — see design.md §12): the
      probe fix and the lighter real pair (`qwen2.5-0.5b` + `tinyllama-chat`)
      were re-run end-to-end against the actual shipped code on the same
      dev GPU used in §6, confirming the fix and reproducing §6 finding 5's
      result. The tight DEC-046 pair was also re-run; it still does not fit
      on this 8 GiB card, now for a clean, diagnosable reason (design.md
      §12) rather than the probe bug — not a regression, and not something
      any code change was made in response to.

## 5. Validation

- [x] 5.1 `uv run pytest tests/unit -q` — 550 passed, 1 xfailed (pre-existing
      xfail, unrelated to this change).
- [x] 5.2 `uv run ruff check .` — all checks passed.
- [x] 5.3 `uv run mypy src/` — no issues found (51 source files).
- [x] 5.4 `openspec validate split-multi-model-serving --strict` and
      `openspec validate --all --strict` — both pass (re-run after every
      doc edit in this section).
- [x] 5.5 Final diff/scope review: confirmed no changes to
      `routing/admission.py`, `core/settings.py`'s admission fields, B4's
      semaphore/waiter mechanism, or B5's batch-priority mechanism; no
      changes outside the files this change's design.md/proposal.md named
      as in scope.

## 6. Documentation / archive

- [x] 6.1 New `docs/DECISIONS.md` DEC-059 entry recording the chosen
      architecture and why, mirroring DEC-058's own decision-record
      structure.
- [x] 6.2 `docs/PHASE-A-ARCHITECTURE.md` §10 roadmap line updated to mark
      B6 complete.
- [x] 6.3 `README.md`, `playground/README.md`, `scripts/dev.sh`,
      `.env.example`, `CHANGELOG.md`, `src/inference_x/core/settings.py`'s
      `INFERENCE_X_LOADED_MODELS` comment, and
      `src/inference_x/benchmarks/hardware.py`'s stale docstring
      cross-reference updated to describe one-model-per-process instead of
      the retired in-process multi-model pattern.
      `docs/UNDERSTANDING-INFERENCE-X.md` deliberately left unchanged — it
      is a dated, point-in-time reverse-engineering snapshot ("as of commit
      `90350d9`"), not a maintained living reference; no prior change (B1-B5)
      updated it either.
- [x] 6.4 Archive this change only after `rescope-admission-control` (B4)
      and `add-batch-priority-queueing` (B5) have themselves been archived.
      Both archived in this Phase-B closure pass, in that order; this
      change follows.
