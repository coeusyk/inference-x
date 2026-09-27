# InferenceX — Architecture Evolution (v0.1 → v1.0)

Engineering source material, not an article. Every `[HISTORICAL]` / `[CURRENT]`
statement below was checked against the repository on **2026-09-26**
(`main` = `c104a1a`, v0.6.0; `develop` = `74e39d5`). Where the approved v1.0
planning source disagreed with the repo, the repo wins and the correction is
listed in §14.

Tags:

- `[HISTORICAL]` — happened; evidence cited (tag, commit, DEC, archived change).
- `[CURRENT]` — true of `develop` today.
- `[PROPOSED]` — approved v1.0 direction, not yet built.
- `[OPEN]` — unresolved question; needs evidence or an owner decision.
- `[DEFERRED]` — deliberately not being done, with the reason.

---

## 0. Purpose

A durable record of why InferenceX's architecture changed, so a reader can
reconstruct the path without rereading all of `docs/DECISIONS.md` (DEC-000 …
DEC-062) and the 30+ archived OpenSpec changes. It is the factual baseline for
a future article; it is not that article.

## 1. Original project

`[HISTORICAL]` First commit 2026-06-07 (`d342bb9`); Phase 1 slice the same day
(`fc1ee1d`). The Phase 1 README described "a self-hosted LLM inference platform
built incrementally on top of vLLM", starting from "a stable OpenAI-compatible
chat completions API" and expanding through registry/routing, observability,
playground/evaluation, and hardening (Phases 1–5, `docs/PHASES.md`).

- DEC-001: one vLLM-backed OpenAI-compatible chat endpoint before routing,
  observability, or UI.
- DEC-002: WSL2 Ubuntu is the primary runtime.
- DEC-020 / DEC-024 / DEC-026: a stdlib CLI client, then a Textual TUI
  playground with model selection — a daily-use interaction path existed from
  early on, not just an evaluation harness.
- DEC-DEFER-01: no auth; server binds loopback for single-user local use.

`[OPEN]` The "learning project" motive is owner-stated; the repo documents the
project as a platform built incrementally, and does not itself say "learning".

## 2. v0.1 → v0.2 era: making it fit consumer GPUs

`[HISTORICAL]` Tags: v0.1.0 (2026-06-25), v0.1.1, v0.1.2 (2026-06-26),
**v0.2.0 (2026-07-05)**. The v0.1.2 → v0.2.0 range (Phases 7–12 in
`docs/PHASES.md`) added:

- Quant-aware VRAM sizing, VRAM tiers (`config/vram_tiers.yaml`: 6gb/12gb/24gb),
  `gpu_memory_utilization: auto` (DEC-035), metrics wiring.
- Pre-dispatch admission control for context length and KV pressure (Phase 8,
  DEC-038).
- A shared per-engine driver thread for non-streaming concurrency (Phase 9;
  later deleted — §4).
- Tier-derived vLLM knobs (`max_num_seqs` etc., Phase 10).
- Load-time model variant selection by VRAM fit, including resolving
  `INFERENCE_X_DEFAULT_MODEL` family names through the selector (Phases 11–12).
- Registered models from `opt-125m` up to `qwen2.5-7b` (bf16 and AWQ variants);
  `qwen1.5-1.8b` added as a "validated 2B-class" entry.

Earlier, Phase 3 (DEC-021) had introduced `EnginePool`: several models in one
process, driven by the TUI's side-by-side compare mode. That decision is
reversed in §4.

DEC-025 (Phase 5 hardening) set the request caps that are still live today:
message `content` ≤ 32,000 chars, ≤ 50 messages, `max_tokens` ≤ 4096.

## 3. Phase A: truthfulness

`[HISTORICAL]` Closed by **v0.3.0 (2026-08-05)**, "Truthful Streaming Metrics,
Deterministic Seeds & Effective-Request Transparency". (A release prep
initially numbered it v0.2.0; commit `51c764f` corrected it to v0.3.0.)

Themes, each with its decision:

- Engine-accounted streamed usage; pre-OS-2 streamed token counts declared
  incomparable (DEC-049, DEC-050).
- Request `seed` and the deterministic-generation contract (DEC-051).
- `strict` may only turn a substitution into a rejection (DEC-052).
- `resolved` + `warnings`: every response reports what was actually run;
  pre- vs post-generation metadata lifecycle (DEC-053).
- Benchmark suite identity as a pinned content hash; `suite_version` is
  necessary but not sufficient for comparability (DEC-054, DEC-055).
- Advisor scores only measured quantities; the VRAM figure is renamed
  `vram_device_occupied_gib` to say what it is (DEC-056, DEC-057).
- An admission reservation leak and a type-unsound engine contract found in a
  pre-Phase-B audit were fixed (archived `fix-admission-reservation-leak`,
  `close-generate-stream-contract-gap`).
- DEC-047 (Engine Boundary and backend plurality): what stays shared versus
  what is backend-specific. It is the constraint every later backend change
  must respect.

The principle that came out of it: **never silently claim behavior that did not
actually occur.**

## 4. Phase B: removing self-imposed bottlenecks

`[HISTORICAL]` Released as v0.4.0 (2026-08-06), v0.5.0 (2026-08-06), and
**v0.6.0 (2026-08-10)**. Roadmap items B1–B6 come from
`docs/REVIEW-2026-08-03-architecture.md` §9. Each one replaced an earlier
design, and the records give the reason:

- **B1 → v0.4.0, DEC-058.** `LLM` + hand-rolled `EngineDriver` → `AsyncLLM`.
  *Why the old design was wrong:* disconnect/timeout only stopped *waiting*,
  while the GPU kept computing. `AsyncLLM` gives a real engine-side abort and
  deletes the Phase 9 driver thread and its race class.
- **B2 → v0.4.0.** `GET /metrics` passes vLLM's own `vllm:` Prometheus series
  through (queue depth, KV-cache utilization, per-phase timing) instead of
  re-deriving them.
- **B3 → v0.5.0.** Per-request `timing` (`queue_time_ms`, `prefill_time_ms`,
  `decode_time_ms`, `inference_time_ms`), derived from the engine's own
  `queued_ts` / `scheduled_ts` / `first_token_ts` / `last_token_ts`. It is
  `None` rather than estimated when the engine can't supply it.
- **B4 → v0.6.0, DEC-060.** Gate 1's instant sequence-concurrency 429 → a
  bounded per-model semaphore wait. *Why:* the old admission layer
  duplicated work the scheduler already does and produced wrong 429s.
- **B5 → v0.6.0, DEC-061.** `priority: batch` waits longer
  (`batch_admission_wait_s`, default 30s) on the *same* FIFO semaphore, with a
  bounded batch-waiter cap (8 × `max_num_seqs`). A priority-preemptive dual
  queue and reserved interactive permits were both rejected (§10.2). The
  evidence was `opt-125m` with `max_num_seqs=4` only.
- **B6 → v0.6.0, DEC-059.** One model per OS process; `EnginePool` refuses
  more than one distinct model; the compare TUI orchestrates one process per
  model. *Why:* in-process multi-model (DEC-021) was "a demo feature charging
  architectural rent" — `enforce_eager` coupling, a `max_model_len` 2048
  clamp, and sequential-VRAM heuristics paid for the whole session. Open,
  non-blocking follow-ups recorded at the time: crash isolation, and the root
  cause of the free-VRAM probe bug.

## 5. Phase C: verifiable inference

`[HISTORICAL]` / `[CURRENT]` Phase C reframed InferenceX toward "a third party,
given only the output artifact, can tell what computation produced a result"
(DEC-062 context; architecture review §8). All of it is on `develop`, **after
v0.6.0, and unreleased**:

- **C1, `add-run-manifest` (DEC-062).** Content-addressed `run_id` (a sha256
  over a canonical preimage — a comparability guarantee, not a signature).
  `X-Run-Id` goes on non-streaming responses, and the body carries `run_id`.
  The original roadmap line included `GET /v1/manifest`; that was not built.
  Manifest *delivery* beyond `X-Run-Id` was left `[OPEN]` as C1 design D5.
- **C3 + C4b, `add-deterministic-execution`.** Deterministic mode and
  `verify-determinism`, including refusing late per-request activation rather
  than lying (`98945c9`).
- **C4a, `add-oracle-conformance`.** Gated teacher-forced conformance against
  `transformers`.
- **C6, `add-plan-doctor`.** `GET /v1/plan`, `GET /v1/doctor`, and
  `make plan` / `make doctor`.
- **C5, Varex integration.** `[CURRENT]` **Not merged.** PR #38
  (`feat/integrate-varex-manifest`, "C5 pt.1") is **open**. It resolves D5 by
  putting the full `manifest` in *every* non-streaming response body. Its
  OpenSpec change (`integrate-varex-manifest`) exists only on that branch.
  Nothing on `develop` returns a body `manifest`.
- **C2, co-batch provenance.** `[DEFERRED]` until C5 shows value (DEC-062).

## 6. What Phase C revealed

`[CURRENT]` The provenance and determinism work made InferenceX much better at
*explaining* inference, but no better as a *daily-use server*. As of `develop`
today:

- **Model fixed at process start.** `INFERENCE_X_DEFAULT_MODEL` /
  `INFERENCE_X_LOADED_MODELS`; more than one distinct model is rejected
  (`api/deps.py`). There is no runtime load, unload, or swap.
- **vLLM only.** `engines/` contains just `vllm_engine.py`.
- **No operator CLI.** There is no `[project.scripts]` entry point; operation
  is Makefile targets plus environment variables.
- **Thin OpenAI compatibility:**
  - The request schema doesn't forbid unknown fields, so `stop` and other
    OpenAI parameters are **silently dropped** by pydantic. That contradicts
    the Phase A principle.
  - DEC-025 caps (32k chars, 50 messages, `max_tokens` ≤ 4096) and roles
    limited to `system` / `user` / `assistant`.
- **Low-VRAM envelope.** On the 6gb tier the validated vLLM envelope is
  roughly 2B-class models. `qwen2.5-7b` is registered as bf16 and AWQ
  variants, but bf16 needs about 13 GiB and AWQ failed to load on the 8 GiB
  dev box (2026-07-02, note in `config/models.yaml`). There is no GGUF path.
- **No validation with a real coding client** (Aider or Continue).

## 7. Varex's role

`[PROPOSED]` Varex stays a downstream consumer and validation workload that
benefits from provenance. It does not define the core product architecture.
Varex validation (V1-5) runs in parallel and never gates V1-0 … V1-4.

## 8. Why Ollama is the comparison

`[PROPOSED]` The comparison is operational: *can InferenceX become the local
server I actually point tools at?* It is not a claim to beat Ollama in general.
The intended differentiators, all of which already exist in some form:

- explicit `resolved`
- `warnings` instead of silent degradation
- `plan` / `doctor`
- native metrics and engine timing
- `run_id` provenance
- backend choice driven by hardware reality

No comparative performance claim is made without measurements.

## 9. v1.0 thesis

`[PROPOSED]` One OpenAI-compatible endpoint serving multiple registered models,
with on-demand model lifecycle across vLLM and llama.cpp (external
`llama-server`), realistic on consumer hardware. The Phase A–C truthfulness,
observability, and provenance machinery is kept as the differentiator.

The v1.0.0 gate is the statement *"I can run InferenceX as my local server
instead of Ollama for my supported workflow"*, backed by real Aider and
Continue sessions, not by task completion.

## 10. Required changes

### 10.1 Sequence `[PROPOSED]`

```text
V1-0  Aider-compatible serving surface (+ opt-in full manifest)
  |
      streaming tool calling for Continue (done, #41)
  |
      vLLM 0.30.0 qualification (done, GO, #44; runtime baseline, not a feature)
  |
V1-1  llama.cpp backend (external llama-server, behind the Engine Boundary)
  |
V1-2  model lifecycle supervisor (above B6's one-model-per-process)
  |
V1-3  `inferencex` operator CLI (serve / ps / plan / doctor / lifecycle)
  |
V1-4  validation + hardening (realistic models, crash isolation, retuned admission)
  |
v1.0.0
V1-5  Varex consumer validation — parallel, non-gating
```

The vLLM 0.30.0 qualification finished on 2026-09-27 with a GO, and #44 pins vLLM to exactly 0.30.0 as the v1.0 runtime baseline. It was run as its own compatibility change, not combined with any feature. The matrix, the performance baseline against 0.22.1 and the regressions it found live in the `qualify-vllm-0-30` OpenSpec change and in DEC-066 (the decision that makes 0.30.0 the exactly pinned, qualified baseline). Admission restoration stays deferred under the triggers in 10.3; finishing the qualification satisfies only the precondition there, not a trigger. V1-1, the llama.cpp backend, is the next core milestone.

Invariants carried forward:

- DEC-047 Engine Boundary
- B6 one model per process
- `AsyncLLM`
- native metrics passthrough
- `EngineTiming`
- `run_id` / `X-Run-Id`
- `strict` / `warnings` / `resolved`
- never fabricate

### 10.2 Manifest delivery `[PROPOSED]` (owner decision, 2026-09-26)

- The full `manifest` becomes **opt-in**.
- `timing`, `resolved`, `warnings`, and `run_id` **stay in the default
  response**; they are existing additive extensions and part of the
  never-lie contract.
- `X-Run-Id` is preserved.
- If Aider or Continue show that one of those fields actually breaks a
  client, that becomes a separate, evidence-backed decision.
- PR #38 is superseded, not merged. Its "always inline" design is exactly
  the part being reversed. Its reconstruction test (recomputing `run_id` from
  the returned manifest) is kept.

### 10.3 Admission restoration `[DEFERRED]` (owner decision, 2026-09-27)

"Admission restoration" is the question of how quickly an interactive request gets served when batch work already occupies the GPU. It is deferred, not solved. Nothing below changes current behavior.

#### The current admission decision

DEC-061 is the decision record for how InferenceX admits requests today. Each model has one shared concurrency pool, sized to the number of sequences the engine runs at once. Requests enter it first come, first served. An `interactive` request waits for a slot up to a short bound (`admission_wait_s`, about 5 s) and a `batch` request up to a longer one (`batch_admission_wait_s`, 30 s); either one that times out gets a 429 with `Retry-After`. The number of queued `batch` requests is capped, and a `batch` request over the cap is rejected at once. A fixed reservation of slots for interactive traffic was considered and rejected, because idle reserved slots waste capacity and there was no evidence for how many to reserve. DEC-061 records its own known cost: an interactive request that arrives behind a batch flood can still wait.

#### What vLLM's priority scheduling actually does (verified from source)

Checked in both the pinned vLLM 0.22.1 and the released vLLM 0.30.0 (2026-09-22); the behavior is the same in both:

- `priority` can reorder requests that are already waiting inside vLLM. Lower values are served first, with ties broken by arrival time.
- A newly arrived high-priority request does **not** evict running work just to start sooner. When all sequence slots are busy or cache space is short, vLLM's waiting-queue loop simply stops for that step.
- When a running request needs more KV cache and none is free, vLLM evicts (preempts) a running request. Under the priority policy the victim is the lowest-priority running request; under the default policy it is the most recently added one.
- Neither version has a `max_num_active_seqs` option. That option exists only on vLLM `main`, added by PR #56758 (merged 2026-09-17, after the 0.30 release branch had diverged). It separates "how many requests may run" from "how large the runner and CUDA graphs are sized". No plan here depends on it being released.

A request's `priority` is still consumed only by `routing/admission.py` and is not passed to vLLM.

#### Why the first proposed experiment was insufficient

The first plan compared two arms:

- **A: current admission plus normal vLLM scheduling.** A valid baseline.
- **B: current admission plus vLLM priority scheduling.** Only a control. InferenceX admission lets no more requests into vLLM than the model's effective concurrency, so the queue forms in InferenceX, in front of vLLM, and vLLM's waiting queue is nearly empty. Its priority policy has almost nothing to reorder. A null result from B would describe that setup; it would not show that vLLM priority is ineffective.

If the work resumes, it needs a third arm:

- **C: an experiment-only admission setting that lets more requests into vLLM than can run at once.** A real waiting queue then exists inside vLLM and its priority policy can act. C is an experiment, **not** an approved production architecture.

Two limits apply even to C:

- Priority does not let a waiting interactive request displace batch work that is already running. Batch generation length, meaning how long running batch work takes to finish, stays a major variable in how quickly service is restored.
- Moving the queue into vLLM bypasses what InferenceX admission provides unless a production design explicitly keeps it: context-length validation, OpenAI-shaped rejections, output-budget resolution, KV-cache token reservation, bounded interactive and batch waits, clean 429 plus `Retry-After`, the batch waiter cap, and truthful `resolved` and `warnings`. vLLM scheduling does not replace InferenceX admission.

#### Why it is deferred

1. It is not a v1.0 product blocker.
2. InferenceX is primarily a single-developer local inference server. Heavy contention between interactive and batch traffic is not a dominant daily workflow today.
3. The large Varex batch use case (Varex is the downstream prompt-evaluation consumer that runs batches of up to about 200 requests) is already non-blocking and deferred (section 11).
4. Restoration measurements are only meaningful on the vLLM version intended for v1.0.
5. Moving from vLLM 0.22.1 to 0.30.0 changes behavior these measurements depend on: a KV-cache watermark meant to reduce preemptions (0.24), a PyTorch upgrade (2.13, in 0.27), a Transformers upgrade (5.15, in 0.28), and a new default model runner, Model Runner V2 (0.29). Measurements taken on 0.22.1 would have to be redone after the upgrade.

#### When it resumes

The investigation resumes only after vLLM 0.30.0 qualification is complete **and** at least one of these is observed:

1. A real interactive request gets a 429 because it timed out waiting behind batch work.
2. Under a real mixed workload, interactive time-to-first-token exceeds 2x the uncontended p95. For example, an evaluation batch is running while Continue or another interactive client is in use.
3. Varex, or another real workload, again needs large concurrent batch traffic.

These are triggers to **investigate**. They are not automatic triggers to change DEC-061's admission policy.

#### Discipline for the future experiment

- Every compared run uses the same qualified vLLM version.
- Admission wait and vLLM queue time are measured separately: admission wait from the client (the stream's first metadata event is sent as soon as admission completes), vLLM queue time from the engine's own timing. They come from different clocks and are not subtracted from each other.
- End-to-end time-to-first-token is a primary, user-visible metric.
- Concurrency is meaningful: not a `max_num_seqs=1` model.
- There is enough KV-cache pressure when testing vLLM's priority-aware preemption path.
- Prefix caching is controlled or prompts are varied, so cache hits do not flatter the results.
- Output budgets are explicit.
- Warm-up, JIT compilation and CUDA-graph capture are excluded from measurements.
- Other GPU processes are stopped.
- Trials are repeated, and medians and p95 are reported.
- Models are realistic, not only `opt-125m`.

Candidate decision thresholds, **an experimental proposal only**, not architecture:

- **Keep DEC-061 as it is** if, in A, no interactive request gets a 429 and interactive p95 time-to-first-token stays within 2x of uncontended.
- **Keep DEC-061 and add vLLM priority** only if C cuts the interactive p95 of admission wait plus vLLM queue time by at least 50%, costs batch throughput no more than 10%, shows no batch starvation, **and** a production design keeps InferenceX admission's guarantees.
- **Reopen DEC-061** if even C does not restore interactive service, because running batch work cannot be displaced.

## 11. Intentionally deferred

`[DEFERRED]`

- C2 co-batch provenance, until there is evidence of value.
- Tools / function calling, until a real acceptance client needs them.
- Distributed or multi-node inference.
- An arbitrary backend ecosystem.
- A Modelfile clone or model marketplace.
- Auth / multi-tenancy (loopback personal use, DEC-DEFER-01).
- A generalized borrow/restore or MoFlux-style capacity framework.
- Batch scaling to ~200 requests (a Varex concern, not a daily-driver one).
- Admission restoration (interactive service under batch contention); see section 10.3 for the verified facts and the resume triggers.
- Deep hardware optimization beyond the v1.0 baseline.

## 12. Known v0.6 limitations → v1.0 treatment

| Limitation | Treatment |
|---|---|
| Admission defaults calibrated on `opt-125m` (DEC-060/061) | Retune on realistic 1–8B models before v1.0 (V1-4) |
| Crash isolation not validated (B6 follow-up) | Validate before v1.0 (V1-2/V1-4) |
| Free-VRAM probe relies on `nvidia-smi` (B6) | OK for vLLM; llama.cpp / CPU / non-NVIDIA must degrade honestly |
| No runtime model switching | v1.0 blocker (V1-2) |
| vLLM only | v1.0 blocker (V1-1) |
| No operator CLI | v1.0 blocker (V1-3) |
| Thin API compatibility / silent field dropping | v1.0 blocker (V1-0) |
| Determinism is vLLM-only and partial | Capability-scoped; unsupported → refuse / report |
| No auth | Accepted for loopback use; documented |

## 13. Framing

InferenceX began as a local, OpenAI-compatible inference-server project. It
accumulated increasingly strong reliability, observability, and provenance
infrastructure. It then leaned for a while toward a benchmark and
verifiable-inference identity, and used what that work revealed to re-center
on becoming a genuinely useful local server. v1.0 was not inevitable; this
record should not read as if it were.

## 14. Corrections applied to the planning source (2026-09-26)

- **C5 inline manifest delivery is not merged.** PR #38 was closed unmerged
  on 2026-09-26 as superseded by V1-0's opt-in manifest; `develop` returns no
  body `manifest`.
- **Phase C is unreleased.** C1, C3/C4b, C4a, and C6 are on `develop` after
  v0.6.0; the latest tag is v0.6.0.
- **C1's roadmap `GET /v1/manifest` was never built.** Delivery was left open
  as C1 D5.
- **The dev GPU changed.** B6's experiments ran on an RTX 4060 (8 GiB, 6gb
  tier). Today's machine is an RTX 3060 Laptop (6 GiB). Earlier measurements
  are not directly comparable to new ones.
- **B1 made the Phase 9 driver thread moot.** B1 (DEC-058) deleted it, along
  with the global step lock that B6's roadmap wording still named.
- **Phase B releases mislabeled.** The v0.4.0 and v0.5.0 release-prep commits
  say "Phase A", but their release notes describe Phase B items (B1/B2 and
  B3).

## 15. V1-0 acceptance (2026-09-27) `[CURRENT]`

The Aider exit test was finished on the desktop, not the laptop: RTX 4060 (8 GiB), which still resolves to the 6gb tier because there is no tier between 0 and 10 GiB. vLLM 0.22.1, Python 3.13.2. `qwen2.5-coder-1.5b` composed to 8192 x 1 and vLLM measured 52,384 KV tokens. That number belongs to this card and this model and is not a general capacity figure.

`scripts/aider_acceptance.py` gives Aider 0.86.2 a repo with a buggy `median()` and a test that fails on it, then runs the test after Aider's edit. It passed 7 of 7 runs (4 streaming, 3 `--no-stream`) using the `whole` edit format. A logging proxy showed that Aider sends only `model`, `temperature`, `stream` and role/content messages, so nothing was rejected or dropped. Non-streaming responses resolved `max_tokens` to the remaining context (7394) with no warnings.

While preparing this, a truthfulness bug from before V1-0 was fixed. `VLLMEngine` replaced `max_tokens` with a model's `max_completion_tokens` after `resolved` had been built, so `opt-125m` could run up to 256 tokens while reporting 8. Admission now resolves the cap (design D11). Live on `opt-125m`, `max_tokens: 8` now runs exactly 8 tokens (`finish_reason: length`).
