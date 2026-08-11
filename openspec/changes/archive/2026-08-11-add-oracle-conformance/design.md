# Design — add-oracle-conformance (C4a)

## Context

Authoritative source: `docs/REVIEW-2026-08-03-architecture.md` §6.2b (token-exact forward validation
against a `transformers` oracle). Planning report §4.5 / §9 / §11: C4 splits into C4a (oracle,
independent) and C4b (determinism conformance, ships with C3). C4a introduces the gated real-GPU
test tier that later Phase C/D work reuses. See proposal.md for motivation.

## Goals / Non-Goals

**Goals**
- Teacher-forced top-1 agreement vs `transformers` on `opt-125m` through the Inference-X engine path.
- Near-ties reported by position — never silently counted as agreement.
- Gated GPU tier that cannot poison the GPU-free `checks` job.
- Unit-tested classification logic so the near-tie policy is enforced in CI.

**Non-Goals (deferred)**
- `make verify-determinism` / C4b (one unique sample over N trials) — belongs with C3.
- Expanding the oracle to every registry model or to chat-templated instruct models.
- A public HTTP endpoint that runs the oracle.
- Hardening vLLM/`transformers` numerical parity beyond reporting (no serving-path patches here).

## Decisions

### D1 — Oracle lives under `tests/oracle/`, not `tests/unit/`
`checks` already runs only `uv run pytest tests/unit -q`. Placing the real-GPU suite under
`tests/oracle/` keeps it structurally outside `checks` without changing the CI workflow's test path.
A `@pytest.mark.gpu` marker plus an env opt-in (`INFERENCE_X_RUN_GPU_TESTS=1`) still guards
`pytest tests/oracle` (and a future `pytest tests/`) on CPU hosts: skip with a clear reason rather
than attempting a weight download or engine start.

### D2 — Compare Inference-X `VLLMEngine` prompt logprobs to `transformers` logits
"Through Inference-X" means constructing the same `VLLMEngine` the composition root uses for
`opt-125m`, not a parallel raw-`LLM` script. Teacher-forcing uses vLLM `prompt_logprobs` on the
fixed token sequence (prompt scoring, `max_tokens=1`) so each position's top-1 is recovered without
autoresgressive sampling noise. The `transformers` side runs a single forward pass over the same
token ids. No new `BaseEngine` method is required for C4a — the oracle is a test harness that
exercises the concrete engine the platform already ships (DEC-047: no speculative
backend-neutral IR).

### D3 — Near-tie policy: report, do not silently pass
A position is a **near-tie** when the margin between the top-1 and top-2 logits (on either side, or
both) is below a small epsilon (default `1e-3` in logit space), or when the two sides disagree on
top-1 but each side's own top-1/top-2 margin is below epsilon. Near-ties are listed in the oracle
report by absolute position and token ids. They do **not** count toward the exact-agreement tally
and do **not** fail the suite by themselves. A **hard mismatch** (disagreement without a near-tie
margin) fails the suite. Pass criterion: zero hard mismatches; exact count and near-tie list are
always printed.

### D4 — Fixed short prompt, causal LM path only
`opt-125m` is not instruction-tuned (`instruction_tuned: false` in `models.yaml`). The oracle uses
a short fixed plaintext prompt (no chat template), identical tokenization on both sides via the
model's tokenizer. This avoids chat-template drift and keeps the first oracle focused on forward
numerics.

### D5 — Pure comparison helpers are unit-tested under `tests/unit/`
Classification (`exact` / `near_tie` / `mismatch`) is a pure function over per-position top-1/top-2
ids and margins. Unit tests feed synthetic tables and assert reporting behavior so CI enforces the
§6.2b policy even when no GPU is present. The GPU suite calls the same helper.

## Risks / Trade-offs

- **[Risk] vLLM vs `transformers` numerics differ on some positions even on a tiny model.** →
  Mitigation: near-tie reporting (D3); fail only on hard mismatches; keep the prompt short and
  greedy/scoring-only (no sampling).
- **[Risk] GPU suite accidentally collected by a broad pytest invocation.** → Mitigation: D1 marker
  + env gate; document `make oracle` / env flag; leave `checks` on `tests/unit` only.
- **[Risk] First run downloads `facebook/opt-125m`.** → Mitigation: document prerequisite; skip
  cleanly when env gate is off so CI never downloads.

## Migration Plan

Additive only. No rollback beyond deleting `tests/oracle/` and the marker registration. No API
migration.
