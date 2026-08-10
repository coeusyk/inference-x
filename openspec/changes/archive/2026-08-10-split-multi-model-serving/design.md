# Design: split-multi-model-serving (B6)

Evidence tags used throughout: **[verified]** (read directly from source in
this repo or its installed dependencies), **[source-derived]** (a direct,
low-inference reading of a decision/review doc already in this repo),
**[inference]** (a reasoned conclusion from verified facts, not itself
directly observed), **[unverified]** (a claim this investigation could not
confirm from static reading and would need a live experiment to settle).

## 1. Current multi-model serving behavior

**[verified]** `EnginePool` (`src/inference_x/engines/pool.py`, 55 lines) is
a thin, static dispatcher:

```python
class EnginePool:
    def __init__(self, engines: dict[str, BaseEngine]) -> None:
        if not engines:
            raise ValueError("EnginePool requires at least one engine")
        self._engines: dict[str, BaseEngine] = engines
```

- No `add`/`remove`/`replace` method exists. The `{name: engine}` map is
  fixed for the process's lifetime — set once at `_build_engine_pool()`
  (`api/deps.py`), never mutated after.
- `get(name)` raises `ValueError` for an unregistered name. `loaded_models()`
  returns insertion order. `all_healthy()` / `health_status()` report a
  per-model `bool` / `"ok" | "unavailable"` by calling each engine's own
  `is_healthy()` — already per-model granular today, not an all-or-nothing
  signal.
- `shutdown()` iterates every engine and calls its `shutdown()` if present,
  swallowing and logging (not raising) any individual failure.

**[verified]** Construction (`api/deps.py:_build_engine_pool`, cached via
`@lru_cache(maxsize=1)`): `pool_size = len(resolved_models)` — the count of
*distinct model names* in `INFERENCE_X_LOADED_MODELS` (comma-separated,
falls back to `default_model` alone). For each model, `VLLMEngine(...)` is
constructed with `pool_size`, the full `pool_models` list, every model's
`pool_configs`, and its own `engine_index`, so every engine in the pool is
aware of the whole pool's shape at construction time — this is exactly the
information `utils/vllm_pool_config.py`'s sequential-VRAM heuristics need
(§2).

**[verified]** `pool_size` counts distinct model *names*, never same-model
replicas — `EnginePool._engines` is a strict 1:1 name→engine map. This was
independently confirmed by both B4's and B5's own investigations
(`rescope-admission-control/design.md` "Semaphore ownership and keying"),
carried forward here rather than re-derived: horizontal replication of one
model is not what `pool_size > 1` means today, and nothing in this
investigation changes that meaning.

**[source-derived]** DEC-058 (B1, `AsyncLLM` migration, 2026-08-05),
Decision 3: *"`pool_size > 1`: not guaranteed, not forbidden. Nothing
verified... demonstrates `AsyncLLM` cannot safely support multiple
co-located instances, so B1 adds no new construction-time validation for
`pool_size` in either direction. B6 remains the only phase authorized to
redesign multi-engine serving."* This is the open question B6 exists to
close — B1 deliberately left it open rather than guessing.

## 2. Step lock, `enforce_eager`, and the pool/VRAM constraints

**Step lock — already deleted, not B6's to remove.** **[verified]** `grep
-rn 'Lock\|_STEP_LOCK\|_POOL_STEP_LOCK' src/inference_x/` finds exactly two
unrelated `threading.Lock` instances (`routing/admission.py`'s
`_PerModelCounter`, `observability/storage.py`) and zero matches in
`engines/`. **[source-derived]** DEC-058 Decision 1 confirms this was
deliberate: *"`_POOL_STEP_LOCK` deleted, not relocated. Each `AsyncLLM`
instance owns an independent background engine-core *process*
(`async_llm.py:146`), not a shared in-process step loop — there is no
shared resource left to guard."* The original roadmap sentence ("delete the
global step lock") predates B1 and is now stale; carrying it into this
change's problem statement unmodified would misdescribe current reality
(this is exactly why `proposal.md` corrects it rather than repeating it).

**[source-derived, with one open caveat]** DEC-058 Decision 1 also states
this deletion *"does not prove multiple co-located `AsyncLLM` instances are
safe, only that the specific mechanism the old lock guarded against has no
equivalent here."* That caveat is still open — see §5 and §11.

**`enforce_eager` — real, current, still B6's to resolve.** **[verified]**
`engines/vllm_engine.py` (construction kwargs, ~line 435):

```python
if pool_size > 1:
    kwargs["enforce_eager"] = True
```

Unconditional on model identity — every engine in a `pool_size > 1` pool
gets CUDA graphs disabled, not just the ones the user is actively comparing
in a given request. **[source-derived]** `REVIEW-2026-08-03-architecture.md`
§3.5 names this "measurably slower decode" but the review does not itself
carry a number, and this investigation ran no experiment to measure it —
the claim "CUDA graphs off is slower" is well-established vLLM behavior in
general, not something re-verified against this specific repo's models this
round; treat the *direction* as verified and the *magnitude* as
**[unverified]**.

**`max_model_len` 2048 clamp — real, current, still B6's to resolve.**
**[verified]** `utils/vllm_pool_config.py:scale_model_config_for_pool`:

```python
else:  # pool_size > 1
    util = _weight_scaled_utilization(...)
    util = _apply_sequential_vram_caps(...)
    if "max_model_len" in scaled:
        scaled["max_model_len"] = min(int(scaled["max_model_len"]), 2048)
```

This overrides *every* tier's `max_model_len_cap` (up to `8192` on the
24gb tier, `config/vram_tiers.yaml`) down to `2048` whenever 2+ models
share a process, silently, regardless of what the YAML or the resolved tier
say elsewhere. A model configured and tier-approved for an 8k context is
capped to 2k context purely because a second model is also loaded — not
because either model individually needs the cap.

**The VRAM sequential-loading heuristics — real, current, ~250-300 LOC,
still B6's to resolve.** **[verified]** `utils/vllm_pool_config.py` (558
LOC total) contains, specifically gated on `pool_size > 1` or
`len(model_configs) > 1`:
- `_weight_scaled_utilization` — splits GPU-memory-utilization budget across
  pool members by estimated weight footprint share.
- `_apply_sequential_vram_caps` / `_multi_engine_overhead_gib` — reserves a
  flat `0.6 GiB` (`_MULTI_ENGINE_OVERHEAD_GIB`, recalibrated by DEC-046
  after it was originally a flat `~3.35 GiB` that broke a real, fitting
  compare pair) per sequential engine-load transition, on top of each
  engine's own footprint.
- `validate_pool_fits` — an aggregate pool-level pre-flight check (sum of
  footprints vs. `total_vram_gib * _POOL_GPU_HEADROOM`), only run when
  `len(model_configs) > 1`.
- `_ARCH_OVERHEAD_GIB` (DEC-045) — a hand-maintained per-architecture
  overhead table used by `estimate_weight_gib`'s embedding/lm_head
  splitting, needed because AWQ/GPTQ quantization doesn't touch embedding
  layers uniformly.

This machinery exists *because* multiple `AsyncLLM` engines currently
initialize sequentially inside one process and must not jointly over-commit
one shared GPU-memory budget that vLLM's own per-engine `gpu_memory_utilization`
knob has no cross-engine awareness of. Removing the `pool_size > 1` case
removes the reason this machinery needs to run at engine-construction time
at all — `_single_engine_utilization` (the `pool_size <= 1` path) is
materially simpler and already the "steady state" every engine would use
under one-model-per-process.

**New, not in the original roadmap wording: Prometheus metrics-registry
collision.** **[verified]**, `engines/vllm_engine.py` (~line 442) and
`CHANGELOG.md`: *"Running with `pool_size > 1`... now logs a startup
warning that each engine's native Prometheus stat logger shares one
process-wide metrics registry, so `/metrics` labels may collide across
engines. This is a recorded, known limitation, not a new restriction."*
**[source-derived]** `expose-native-engine-metrics/design.md`'s own
Non-Goals section states explicitly: *"Solving the `pool_size > 1`
label-collision risk — this change records it (a startup warning), it does
not guard against it. Multi-engine serving redesign is B6's scope."* This
was introduced by B2 (native Prometheus passthrough, already merged and
archived) and explicitly deferred to this change — it belongs in B6's
problem statement even though the original 2026-08-03 roadmap sentence
predates B2 and doesn't mention it.

## 3. Implications of moving multi-model serving across process boundaries

**What changes if each model gets its own OS process:**

- **GPU-memory budgeting simplifies from cross-engine to per-engine.** No
  process-boundary knowledge of sibling models is needed at construction
  time — `_single_engine_utilization` (already the `pool_size <= 1` path
  today) becomes the only path. `validate_pool_fits`'s aggregate multi-model
  check disappears with it — see §6 for what that check currently catches
  and what would replace it.
- **CUDA graphs can be re-enabled per engine** (`enforce_eager` no longer
  forced), at the cost of each engine's own CUDA-graph-capture VRAM
  overhead (`_CUDAGRAPH_OVERHEAD_GIB = 0.45` is already a named constant in
  this file, currently used for the *disabled* (`pool_size == 1`) case's
  footprint estimate — meaning today's single-engine footprint estimate
  already budgets for graphs being on; a `pool_size > 1` engine today
  *doesn't* spend this VRAM because graphs are forced off, so re-enabling
  them for what are today "pool" engines is a real, currently-unspent VRAM
  cost, not free — see §6).
- **Each model's `max_model_len` can use its tier's real cap** — no
  process-shared clamp.
- **Prometheus registry collision disappears** — one process, one model, one
  registry, by construction.
- **Process lifecycle becomes per-model.** Starting, stopping, restarting,
  or crashing one model's server no longer has any code-level relationship
  to any other model's server. `EnginePool` as a construct becomes
  unnecessary (or degenerates to always holding exactly one engine) since
  there is no longer an in-process multi-model dispatch problem for it to
  solve.
- **A new problem is created: cross-process orchestration.** Something now
  has to (a) decide how many processes to start, (b) pick non-colliding
  ports, (c) start them, (d) know when each is healthy, (e) stop them
  together on request, (f) — for the TUI specifically — route each compare
  panel's requests to the right process. None of this exists today for the
  TUI (`playground/app.py`); it does already exist, partially, for the
  batch client (§4).
- **The pre-flight "will this combination even fit" check is lost** unless
  deliberately replaced. Today, `validate_pool_fits()` runs once, before any
  engine loads, and rejects an impossible combination with one clear error
  naming every model. Under N independent processes, the natural failure
  mode is "the second process's own single-engine VRAM probe fails after
  the first process has already claimed its share" — a worse, later, and
  per-process-only error unless an external orchestrator does the
  aggregate check itself first (see §11).

## 4. Existing evidence this pattern already works

**[verified]** `playground/client.py` (the batch CLI, not the TUI) already
implements dual-process compare:

```
python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
    --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
    "Write a haiku about GPUs."
```

— two already-running, independently-started server instances, addressed
purely over HTTP, plus a documented `--sequential` fallback ("Compare on
single GPU (restart server between models)") for hardware that can't hold
two full engines at once. This is exactly the target pattern §3 describes,
already shipped and presumably exercised, in the one InferenceX consumer
that isn't the TUI.

**[verified]** `playground/app.py` (the TUI, `make playground`) does **not**
do this yet. It takes exactly one `base_url` (`DEFAULT_BASE_URL =
"http://localhost:8000"`, a single `--base-url` CLI flag) and its "compare"
mode (`ModelSelectScreen(self.base_url, mode="compare")`) picks two *models*
against that *one* server — i.e. it is the consumer that actually drives
the `pool_size > 1` / `EnginePool` path this change would retire. Closing
this gap (giving the TUI the same dual-base-url capability
`playground/client.py` already has) is necessary work under the
recommended direction, not optional polish — without it, `make playground`
compare mode would break.

**[verified]** Both `playground/app.py` and `playground/client.py` import
no server-side code (`grep -rn 'from inference_x\|import inference_x'
playground/` returns nothing) — confirming `REVIEW-2026-08-03-architecture.md`
§3.5's "the TUI is already a pure HTTP client" claim. Whatever B6 changes on
the server side, the TUI/client only ever need a base URL to talk to it —
there is no shared-process coupling to break.

## 5. Failure and isolation semantics

**[source-derived]** Today, a single model's engine entering a dead/errored
state is already independently observable: `EnginePool.health_status()`
reports per-model `"ok" | "unavailable"` by calling each engine's own
`is_healthy()`, and DEC-058 confirms `AsyncLLM.errored` / `EngineDeadError`
are real, per-engine signals `VLLMEngine` already surfaces — not a
process-wide flag. **[inference]** Since each `AsyncLLM` instance already
runs its own engine-core as a separate OS *process* (not merely a thread) —
this was DEC-058's own justification for deleting the step lock — a crash
*inside* one model's vLLM engine core today likely already surfaces as that
one engine's `errored`/`dead_error` state rather than crashing the parent
FastAPI/uvicorn process, without needing B6 to add anything. This is
**[unverified]**, not confirmed by a live crash test in this investigation
or (as far as this investigation found) in any prior one — it is inferred
from DEC-058's architecture description, not independently reproduced.

**What one-model-per-process would add on top of that:** isolation against
failure modes that are *not* per-engine-core-process today — most notably
an OS-level event (host OOM-killer targeting the whole `uvicorn` process
because one model's memory use grew unboundedly, a driver-level GPU reset,
a Python-level unhandled exception or deadlock in FastAPI/uvicorn itself)
would currently take every loaded model down together; under N processes it
would only take down the one process. **[inference]** This is a real,
directionally-correct isolation gain, but this investigation did not
attempt to reproduce a host-level OOM or driver-panic scenario to confirm
today's single-process blast radius empirically — the claim is architectural
reasoning, not a reproduced failure.

## 6. Resource and GPU/VRAM implications — live-experiment findings (this round)

**Experiment run this round, on this repo's actual dev hardware** (RTX 4060,
8188 MiB total, confirmed via `nvidia-smi` — resolves to the `6gb` tier per
`config/vram_tiers.yaml`'s `min_vram_gb` thresholds), against real, cached
models (`qwen2.5-0.5b`, `qwen2.5-1.5b`, `tinyllama-chat`), via
`uv run uvicorn` server processes started and measured with `nvidia-smi`,
config-only (no `src/`/`tests`/`pyproject.toml` change). Four sub-experiments:

**1. Baseline (today's pool path, `pool_size=2`, forced `enforce_eager`),
the exact DEC-046 reference pair (`qwen2.5-0.5b` + `qwen2.5-1.5b`) — FAILED.**
**[verified]** The second engine's construction raised `RuntimeError: vLLM
engine subprocess failed for qwen2.5-1.5b. GPU has 6.93/8.00 GiB free.`
The underlying vLLM traceback (captured in full) shows the real cause:
`ValueError: No available memory for the cache blocks` — vLLM's own
`gpu_worker.py` logged `Available KV cache memory: -0.64 GiB`, i.e. a
genuine negative budget, not a vague OOM. **This is a regression against
DEC-046's own recorded result** (that exact pair "live-verified... with
~0.74 GiB still free" on 2026-07-03) — the pool path for this reference
pair does not work today, independent of anything B6 changes.

**2. Two independent processes (Option A's mechanism), the same
`qwen2.5-0.5b` + `qwen2.5-1.5b` pair, graphs on — ALSO FAILED, with the
identical root cause.** **[verified]** Process A (`qwen2.5-0.5b` alone)
succeeded (`utilization=0.299`, 3050 MiB used). Process B
(`qwen2.5-1.5b`, started independently, its own OS process) failed with the
exact same `ValueError: No available memory for the cache blocks` /
`Available KV cache memory: -0.81 GiB`. **Root cause identified and
confirmed [verified], reproduced twice** (immediately, and again after a
deliberate 20-second settling delay before starting process B — see
sub-experiment 3): `utils/vllm_pool_config.py:probe_gpu_memory_gib()`
(`torch.cuda.mem_get_info(0)`, called fresh in the second process) reported
`free=6.93 GiB` — essentially the *idle* card state — even though process A
was already confirmed healthy and independently verified via `nvidia-smi`
to be holding 3050 MiB (real free at that moment: 4907 MiB / 4.79 GiB, not
6.93 GiB). A direct `nvidia-smi` call issued from the same shell script at
the same moment correctly showed the reduced free figure. **This is a
probe-accuracy bug, not a CUDA-graph memory-cost effect** — `_single_engine_
utilization()`'s `free_vram_gib` input was simply wrong.

**3. Settling-delay retest — confirms this is structural, not a timing
race.** **[verified]** Waiting 20 seconds between starting process A and
probing again did not change `torch.cuda.mem_get_info()`'s reading (still
`free=6.93 GiB`, identical to immediately after process A started), while
`nvidia-smi` continued to correctly report the reduced free figure the
whole time. This rules out "just add a delay" as a fix.

**4. Critical finding: this bug is orthogonal to the Option A vs. status-quo
question, and pre-exists this investigation.** **[verified]** The *same*
`free=6.9 GiB` reading, from the *same* `probe_gpu_memory_gib()` call,
appears in sub-experiment 1's (pool-path) log for the second engine, inside
the *same* process's *sequential* `_build_engine_pool` loop — not just
across independent processes. **This means today's shipped
`pool_size > 1` path has exactly the same latent bug** this investigation
found in the two-process case; it is not a risk B6 would introduce, it is a
risk B6's target architecture and the status quo both already carry
equally. Fixing it (e.g., sourcing free-VRAM from an `nvidia-smi` subprocess
query instead of `torch.cuda.mem_get_info()` — observed accurate in this
same experiment, though not implemented or tested as a fix this round)
would benefit either option and is not itself a reason to prefer one over
the other.

**5. A real, successful two-independent-process load, with a lighter pair —
direct positive evidence for Option A's mechanism.** **[verified]**
`qwen2.5-0.5b` (port 8000) + `tinyllama-chat` (port 8001), each its own
process, `enforce_eager` not forced (graphs on): both started successfully
(`utilization=0.299` and `0.399` respectively), and **combined peak VRAM
was 6543 MiB used / 1414 MiB free** on the 8188 MiB card — a real,
comfortable margin, not a near-miss. Both processes correctly served their
one model over HTTP throughout. This is the first concrete, positive,
apples-to-apples demonstration in this investigation that Option A's actual
mechanism — N independent single-engine processes, graphs on, no pool
heuristics — works on this repo's real target hardware for a real
(if lighter-than-DEC-046's) pair.

**What this settles vs. what it doesn't:**
- **Settled:** the free-VRAM-probe bug is a real, pre-existing,
  cross-cutting defect, not an Option-A-specific risk — it does not argue
  against Option A relative to the status quo.
- **Settled:** Option A's core mechanism (independent processes, graphs on)
  demonstrably works today for a real pair with real headroom to spare.
- **Not settled:** the *magnitude* of CUDA-graph-capture's own VRAM cost
  (the original framing of this experiment) — neither configuration
  produced a valid apples-to-apples measurement of *that specific* variable
  for the tight `qwen2.5-0.5b`+`qwen2.5-1.5b` pair, because both failed for
  an unrelated, shared reason before either could be measured at that pair
  size. This narrower question remains open (§11 item 1, revised).
- **Not settled:** whether the DEC-046 regression (finding 1) has some
  other explanation this investigation didn't chase further (e.g. a vLLM
  version change since 2026-07-03) — flagged, not diagnosed further, since
  it is the same bug already isolated in findings 2-4 and diagnosing the
  *why* of the probe inaccuracy itself is out of this investigation's
  no-`src`-change scope.

**[verified]** `validate_pool_fits`'s aggregate pre-flight check (§3) would
no longer run for the multi-process case unless replaced. Given finding 4
above, an external replacement must **not** simply reuse
`probe_gpu_memory_gib()` as-is (it would inherit the same blind spot); it
would need either an `nvidia-smi`-based cross-process-aware probe or
explicit, non-`"auto"`, orchestrator-assigned `gpu_memory_utilization`
values per process. This was an open question in the prior round; it is now
a **concrete, scoped implementation requirement**, not a vague follow-up
(§11 item 3, revised).

## 7. API/runtime compatibility constraints

**[verified]** `BaseEngine` (`engines/base.py`) declares `generate`,
`generate_stream`, `is_healthy`, and the non-abstract `count_prompt_tokens`.
Nothing in this contract references `pool_size`, `EnginePool`, or any
multi-engine concept — the Engine Boundary (DEC-047, "Engine Boundary and
backend plurality") is already expressed entirely in terms of one engine.
**[inference]** This means a one-model-per-process redesign requires zero
changes to `BaseEngine`, `ChatService`, `TaskRouter`, or any route handler
— consistent with how B1 (`migrate-async-llm-engine`) already changed
`VLLMEngine`'s internals with "zero edits... in `api/`, `services/`, or
`routing/`" (`PHASE-A-ARCHITECTURE.md` §"B1 status: complete").

**Correction to this section, made this round after live-checking the
actual route source (the prior round asserted this without reading
`api/routes/models.py` directly — flagged and fixed here rather than left
standing):** **[verified]** `GET /v1/models` (`api/routes/models.py`)
is **not** a function of `EnginePool` at all — it depends on `get_registry`
(`ModelRegistry.all()`), the full static `models.yaml` listing, independent
of what is actually loaded. **Confirmed live**: both independent processes
in this round's experiment (§6, finding 5) returned the identical
eight-model registry listing at `GET /v1/models`, despite each having
exactly one model actually loaded. **This means `GET /v1/models`'s
behavior is unaffected by this change** — it already reports the same
thing today regardless of `pool_size`, and would continue to under
one-model-per-process. §11 item 4 (prior round) is resolved: no
compatibility question exists here.

**[verified]** `GET /health` (`api/routes/health.py`) is the one endpoint
that *does* reflect real pool state — its `loaded_models` field comes from
`ChatService.loaded_models()` → `EnginePool.loaded_models()`. Under
one-model-per-process, each process's `/health` would report exactly its
own one model instead of today's full pool list. This is a real, but now
precisely-scoped, observable behavior change — smaller than the prior
round described (only `/health`, not `/v1/models`) and arguably a more
*accurate* signal (a process truthfully reporting only what it serves), not
obviously something to guard against.

**[source-derived]** DEC-047 reserves authorization for "a second concrete
[inference] backend" and Phase D's engine factory / optional-`vllm` /
`llama-server` proxy work to "its own accepted ADR" — B6 is not that ADR
and must not be read as implicitly authorizing backend plurality; this
change is about process topology for the existing single backend
(`VLLMEngine`/vLLM), not about adding a second backend (§8).

## 8. B4/B5 guarantees that must remain intact

**Established by B4 and B5's own investigations, not re-derived here:**

- **Single `AdmissionController` instance, single event loop, per process**
  (`rescope-admission-control/design.md` "Semaphore ownership and keying").
  This remains true, unchanged, under one-model-per-process — it just means
  each process's single `AdmissionController` now only ever sees requests
  for the one model that process serves, which is a strict simplification
  of its input space, not a new requirement on it.
- **B4's bounded-wait semaphore and B5's priority-differentiated wait /
  batch-waiter cap are both process-local, per-model-keyed state**
  (`_PerModelCounter`, `_semaphores` dict). Both already work correctly
  with exactly one model per process — nothing about their design assumed
  or required `pool_size > 1`; both explicitly named zero dependency on
  B6's process-split work in either direction
  (`add-batch-priority-queueing/design.md` §9 "B6 boundary").
- **Verified this round, not merely repeated:** neither `admission.py` nor
  `chat_service.py` nor `core/settings.py`'s admission settings reference
  `EnginePool`, `pool_size`, or anything in `utils/vllm_pool_config.py`
  (`grep` finds no cross-reference in either direction). This change's
  eventual implementation must keep it that way — admission control is not
  this change's file boundary (§9 "Non-goals").

## 9. Non-goals — explicit boundary against Phase C, Phase D, and unrelated redesign

This change does **not**, and must not be made to:

- Touch `routing/admission.py`, `core/settings.py`'s admission fields,
  `services/chat_service.py`, or any B4/B5 mechanism (§8).
- Implement Phase C: signed run manifests / `X-Run-Id` (C1),
  `batch.co_batched_request_ids` (C2), `deterministic: true` /
  `VLLM_BATCH_INVARIANT=1` (C3), the oracle/conformance suite (C4), a real
  Varex end-to-end run (C5), `make plan`/`make doctor` (C6) —
  `PHASE-A-ARCHITECTURE.md` names Phase B and Phase C as sequenced,
  separate phases; nothing in this investigation found a dependency from
  Phase C onto this change or vice versa.
- Implement Phase D: a second inference backend, an engine factory
  (`create_engine`), optional-`vllm`, or a `llama-server` proxy — DEC-047
  explicitly reserves that to "its own accepted ADR... DEC-047 does not
  authorize it," and this change is not that ADR (§7).
- Redesign the model registry, variant routing (`select_variant`), or the
  benchmark advisor — no evidence found connecting any of them to the
  `pool_size > 1` problem this change addresses.
- Decide or implement backend plurality of any kind — this change is about
  how many *processes* serve the existing single vLLM backend, never about
  a second backend.

**One opportunistic, explicitly-optional item named by a prior audit, not a
requirement of this change:** `docs/REVIEW-2026-08-04-phase-a-final-audit.md`
names **DUP-C2** ("fold the triplicated tier resolution and duplicated VRAM
math into one helper") as work suggested to land "while roadmap B6 rewrites
`_build_engine_pool`" — this is a real, source-named suggestion to bundle
if implementation touches that function anyway, not a scope requirement of
this investigation or a blocking dependency in either direction.

## 10. Recommended direction — sufficiently supported this round; two
concrete implementation prerequisites remain, not architectural blockers

**Option A — one model per OS process, no in-process `EnginePool`
multi-engine path.** Retire the `pool_size > 1` branch entirely: delete
`enforce_eager` coupling, the 2048 clamp, and the multi-engine VRAM
heuristics (§2); `EnginePool` degenerates to (or is replaced by) a
single-engine holder; multi-model serving (TUI compare mode) becomes N
independent server processes on N ports, using the same dual-base-url
pattern `playground/client.py` already proves out (§4), with the TUI
(`playground/app.py`) gaining the base-url-per-panel capability it
currently lacks, plus new orchestration (script or `Makefile` target) to
start/stop N processes as one `make playground` command.

This is the option the architecture review already named
(`REVIEW-2026-08-03-architecture.md` §3.5), the option half-already proven
by `playground/client.py`, and the option that actually deletes the ~250-300
LOC of heuristics + both coupling points + the metrics-collision risk
rather than working around them.

**Revised this round: the conditional framing from the prior investigation
round is resolved, not merely reworded.** §6's live experiment found that
the specific risk this recommendation was made conditional on — CUDA
graphs costing prohibitively more VRAM under Option A than today's
forced-eager pool — could not be cleanly measured for the tight reference
pair (both configurations failed identically, for a shared, pre-existing,
Option-A-independent reason: §6 findings 1-4), **but Option A's actual
mechanism was independently, positively demonstrated to work** on this
repo's real dev hardware for a real pair with real headroom (§6 finding 5).
Nothing found this round argues against Option A relative to the status
quo — the one bug found affects both equally. **Recommendation upgraded
from conditional to supported**, with two concrete, scoped implementation
prerequisites (not open architectural questions):

1. **The free-VRAM probe must not be `torch.cuda.mem_get_info()`-only
   across independent processes** (§6 finding 4) — implementation must use
   an `nvidia-smi`-based or otherwise cross-process-aware probe, or
   explicit non-`"auto"` per-process `gpu_memory_utilization` values. This
   is pre-existing debt this change would be wise to fix, not a new
   requirement Option A specifically introduces.
2. **A pre-flight VRAM check must replace `validate_pool_fits`** (§6, §11
   item 3) using the corrected probe from (1) — not the current
   `probe_gpu_memory_gib()` as-is.

Neither prerequisite is a reason to prefer Option B or C — both would need
the same probe fix if they touched multi-model VRAM sizing at all.

**This investigation is closed.** Nothing above is a call for further
experimentation — it is the recommendation this investigation set out to
produce, now evidence-backed rather than conditional. What remains is a
change-owner decision, not more investigation (`tasks.md` §2): (1) approve
Option A itself, and (2) decide whether prerequisites 1 and 2 above ship
inside this change's implementation or as a separately-scoped dependency.
§11's items 2 and 6 are genuinely open follow-up experiments but are
explicitly non-blocking for either decision.

**Option B — keep the single-process pool, stop forcing the degradations
unconditionally.** Investigate whether `enforce_eager`, the 2048 clamp, and
the sequential-VRAM heuristics are still load-bearing under `AsyncLLM`'s
per-engine-subprocess model (they were designed for a different execution
model, the pre-B1 synchronous `LLM` + `EngineDriver` setup) or were
compensating for a constraint that no longer applies. Rejected as the
primary recommendation: it directly contradicts the roadmap's explicit
process-split wording, does not touch the Prometheus registry-collision
risk (still one process, one registry), and — critically — this
investigation found no verified evidence either way on whether concurrent
CUDA graph capture across co-located `AsyncLLM` instances sharing one
physical GPU is actually safe; recommending it without that answer would
be recommending an unverified risk reduction over a verified, already-
partially-proven one.

**Option C — hybrid, keep `pool_size > 1` as a fallback for VRAM-constrained
tiers.** Rejected as the *default* (defeats the stated purpose of deleting
the heuristics). §6's experiment did not surface a reason to fall back to
this — no evidence found this round that Option A fits worse than the
status quo on the tightest tier; the one failure mode found (§6 findings
1-4) hits Option C's own pool path exactly as hard. Retained only as a
theoretical fallback if a future, more targeted experiment (post the probe
fix in §10) finds a genuine per-tier VRAM shortfall Option A cannot solve.

## 11. Remaining items — resolved this round, revised this round, and still
genuinely open

**Resolved this round (§6, §7):**

1. ~~Real VRAM cost of `enforce_eager=False` on 2+ co-located engines~~ —
   **superseded, not directly answered.** The live experiment found a
   different, more fundamental, shared bug (free-VRAM-probe staleness, §6
   findings 1-4) that blocked a clean measurement of *this specific*
   question for the tight reference pair, but also found (§6 finding 5)
   that Option A's actual mechanism works with real headroom for a lighter
   real pair. This is no longer treated as a blocking unknown (§10) — see
   the narrower open remainder below.
4. ~~Whether `GET /v1/models` reporting one model per process is an
   acceptable API change~~ — **resolved, no concern.** `GET /v1/models` was
   verified this round to already report the full model registry regardless
   of `pool_size`, not the pool's loaded-model list (§7 correction). Only
   `GET /health`'s `loaded_models` field actually changes under
   one-model-per-process, and that is expected, accurate behavior.

**Revised this round, now a concrete implementation requirement rather than
an open question (§6, §10):**

3. **`validate_pool_fits`'s replacement must not reuse
   `probe_gpu_memory_gib()` unmodified** — it inherits the exact bug found
   in finding 4. The replacement needs either an `nvidia-smi`-based
   cross-process-aware probe (observed accurate in this round's experiments
   but not implemented or tested as a fix) or explicit, orchestrator-
   assigned, non-`"auto"` per-process `gpu_memory_utilization` values.

**Still genuinely open — require either a decision or further investigation
this round did not attempt:**

2. **Whether a co-located-`AsyncLLM`-instances crash genuinely stays
   isolated to one engine today** (§5) — DEC-058's architecture description
   implies it should; no live crash/OOM reproduction was attempted this
   round to confirm it empirically. Weak, indirect supporting evidence from
   this round: process B's *startup* failure (§6 finding 2) did not
   destabilize or crash already-running process A in either the two-process
   or pool-sequential case — but a clean startup failure is a different,
   weaker claim than isolation of a mid-flight runtime crash, and this
   investigation does not conflate the two.
5. **Magnitude (not direction) of the `enforce_eager` decode-speed cost**
   (§2) — still not measured against this repo's specific models; the two
   successfully-loaded engines in finding 5 were never sent a generation
   request, so no decode-speed data exists from this round either.
6. **Root cause of `torch.cuda.mem_get_info()`'s inaccuracy on this
   platform** (§6) — reproduced three times, not diagnosed further (WSL2
   GPU-passthrough accounting, a PyTorch/CUDA-driver caching behavior, and
   a vLLM-version change since DEC-046 are all plausible, none confirmed).
   Not needed to act on the finding (§10's prerequisite 1 doesn't require
   knowing *why*, only that `nvidia-smi` was independently observed
   accurate where `torch.cuda.mem_get_info()` was not), but named here as
   a genuine unresolved question, not silently dropped.
7. **Whether the DEC-046 regression (§6 finding 1) has a cause specific to
   this pair/environment beyond the shared probe bug** — not chased further
   this round; the probe bug fully explains the observed failure, so no
   further diagnosis was pursued, but this is not the same as proving no
   second factor exists.

Items 2 and 5 are genuine follow-up experiments, not decisions — they can
be run post-approval, during implementation, without blocking the
architectural decision. Item 6 is optional diagnostic depth. None of these
block change-owner approval of the recommendation in §10.

## 12. Post-implementation verification (live GPU, this round)

After implementation, prerequisite 1 (the probe fix) was verified live on
the same dev GPU (RTX 4060, 8188 MiB) used in §6, against the actual shipped
code — not the experimental script §6 ran.

- **Probe accuracy, confirmed non-stale.** With process A (`qwen2.5-0.5b`)
  already resident and using 3057 MiB, process B's `_build_engine_pool` call
  reported `free=4.7 GiB` — matching `nvidia-smi`'s own concurrent reading
  (4900 MiB / ~4.79 GiB) closely, not the wildly-inflated stale reading §6
  finding 4 documented for the old `torch.cuda.mem_get_info()`-based probe.
- **The lighter pair (§6 finding 5) reproduced end-to-end under the shipped
  code**, not just the experimental script: `qwen2.5-0.5b` + `tinyllama-chat`
  as two independent processes both loaded successfully, both served a real
  `/v1/chat/completions` request with a real generated response and
  `timing` data (closing part of item 5 above — a generation request *was*
  sent this round, though a same-model eager-vs-graphs A/B comparison still
  was not), and `GET /v1/metrics` on each process reported accurate,
  independently-attributed VRAM figures. Final state: 6551 MiB used / 1406
  MiB free of 8188 MiB — matching §6 finding 5's 6543/1414 MiB closely.
- **The tight DEC-046 reference pair (`qwen2.5-0.5b` + `qwen2.5-1.5b`) still
  does not fit on this 8 GiB card — now for a clean, diagnosable reason, not
  a probe bug.** With the accurate 4.7 GiB free reading, sizing computed
  `gpu_memory_utilization=0.575` for `qwen2.5-1.5b`; vLLM's own CUDA graph
  memory profiling (enabled by default, no longer disabled by a forced
  `enforce_eager=True` under Option A) left "Available KV cache memory:
  -1.12 GiB" and the engine correctly refused to start with a clear,
  actionable error rather than a confusing stale-probe failure. This is not
  a regression or an implementation defect — it is direct evidence that, for
  *this specific tight pair*, restoring CUDA graphs (the entire point of
  retiring `enforce_eager` coupling) costs enough VRAM to no longer fit
  where a forced-eager engine might have (barely) fit before. It completes,
  rather than contradicts, the original narrow question this investigation's
  §6 could not cleanly measure (proposal.md Status) — the answer for this
  particular pair on this particular card is "no room to spare," while the
  actually-recommended, lighter pair has real headroom. No implementation
  change was made in response to this finding: `validate_model_fits`
  correctly rejects (or, when construction proceeds regardless because the
  preflight check's own approximation differs slightly from vLLM's exact
  runtime allocation, vLLM's own engine construction still correctly refuses
  and surfaces a clear, actionable error either way) rather than silently
  loading into a state with no KV cache. `make playground`/`make
  playground-compare` do not hardcode a default pair — a user picking a tight
  pair like `qwen2.5-0.5b`+`qwen2.5-1.5b` on an 8 GiB card will see this same
  clear rejection, not a crash or a stale-probe-confused failure.
- **`ensure_models_loaded()` itself — the orchestration function this change
  rewrote most heavily — was exercised end-to-end against real subprocesses**,
  not just through its unit tests' mocked boundaries. This run surfaced one
  real bug in the first cut of `start_playground_server`: it called
  `env.pop("INFERENCE_X_LOADED_MODELS", None)` to clear the var for the
  spawned process, but `core/settings.py`'s `load_dotenv(..., override=False)`
  re-populates any var *absent* from the subprocess's environment from
  `.env` — so a `.env` with a multi-model `INFERENCE_X_LOADED_MODELS` (the
  exact line `.env.example` shipped until this change) would have been
  silently refilled inside every spawned process and rejected by the new
  `_build_engine_pool` check. Fixed by setting
  `env["INFERENCE_X_LOADED_MODELS"] = model` instead of popping it, so the
  single-model value always wins over `override=False`'s dotenv fill-in
  regardless of what `.env` contains. A regression test
  (`test_start_playground_server_sets_loaded_models_to_single_model`) seeds a
  multi-model `INFERENCE_X_LOADED_MODELS` via `monkeypatch.setenv` and asserts
  the subprocess `env` kwarg is forced to the single requested model. With the
  fix in place, `ensure_models_loaded("http://127.0.0.1:8000",
  ["qwen2.5-0.5b", "tinyllama-chat"])` was run directly (not through the TUI)
  against this repo's real `.env` (which sets `INFERENCE_X_DEFAULT_MODEL` but
  not `INFERENCE_X_LOADED_MODELS`): both processes started on ports 8000/8001,
  both reached "Engine ready," `GET /health`'s `loaded_models` on each process
  showed exactly its own model, and the returned `{model: base_url}` mapping
  (the same dict `playground/app.py` assigns directly to `self.base_urls`) was
  correct. `stop_playground_server()` released both processes' VRAM back to
  the pre-run baseline (7576/8188 MiB free before and after).
