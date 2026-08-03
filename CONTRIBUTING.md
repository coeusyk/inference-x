# Contributing to InferenceX

InferenceX is a personal learning project, open to contributions that fit its scope:
self-hosted LLM inference on consumer hardware, with a clean layered API and useful
developer tooling. Contributions are welcome but the bar is specificity — vague
proposals or broad refactors without a concrete problem statement will be closed.

---

## Before you start

Read these first:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layer responsibilities and boundaries
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — why the non-obvious choices were made
- [`docs/PHASES.md`](docs/PHASES.md) — what was built and in what order
- [`AGENTS.md`](AGENTS.md) — repo working agreement (file boundaries, anti-scope rules)

If your change would violate a decision in `DECISIONS.md` or cross a file boundary in
`AGENTS.md`, explain why in your proposal before writing any code.

---

## What fits

- Bug fixes with a clear reproduction case
- WSL2 / consumer GPU compatibility improvements
- New model configurations in `config/models.yaml`
- Observability improvements (new metrics, exporter formats)
- Playground UX fixes (not full redesigns)
- Documentation corrections and clarifications
- Test coverage for untested paths

## What does not fit

- Authentication / multi-user support (DEC-DEFER-01 — out of scope for local-only deployment)
- Rate limiting (DEC-DEFER-02)
- Implementing a second inference backend, or introducing backend-neutral
  packages/contracts (e.g. `inference_x/execution/`), without a dedicated
  accepted change that authorizes a concrete second implementation (DEC-047).
  vLLM is the only supported backend today; backend plurality is a long-term
  architectural direction, not a scheduled deliverable.
  Thin Engine Boundary hygiene (for example, an engine factory in
  `engines/registry.py` and durable capability declarations on `BaseEngine`)
  is in scope when it aligns with the current accepted architecture.
- Breaking changes to the OpenAI-compatible API contract
- New playground tabs or major UI additions without a prior discussion

If you're unsure, open an issue before writing code.

---

## Setup

```bash
git clone https://github.com/coeusyk/inference-x.git
cd inference-x
uv sync
cp .env.example .env
uv run pytest tests/unit -q   # must pass without a GPU
```

> `uv` is required. Do not use bare `pip` or `python` — vLLM and its CUDA wheels are
> managed through the `.venv` created by `uv sync`.

For changes that require a GPU, test on WSL2 Ubuntu. That is the supported runtime.
Non-WSL2 Linux may work but is not the primary target.

---

## Making a change

### 1. Open an issue first (for anything non-trivial)

Describe:
- What problem you're solving
- What you tried and why it didn't work
- What you propose to do

For bugs: include the exact error, the command that produced it, your GPU model, and
the output of `nvidia-smi`.

### 2. One concern per pull request

Keep PRs focused. A bug fix and an unrelated cleanup in the same PR will be asked to
split. The diff should be readable in one sitting.

### 3. Follow the file boundaries

From `AGENTS.md`:

| Path | What belongs there |
|---|---|
| `src/inference_x/api/` | FastAPI routes, deps, error mapping — nothing else |
| `src/inference_x/services/` | Orchestration and use-case logic |
| `src/inference_x/engines/` | Engine interfaces, registry/factory, and concrete inference implementations (vLLM today) |
| `src/inference_x/routing/` | Model selection policies |
| `src/inference_x/observability/` | Middleware, metrics, storage, exporters |
| `src/inference_x/schemas/` | Request/response models only |
| `src/inference_x/benchmarks/` | Benchmark runner, hardware profiler, advisor |
| `playground/` | TUI clients only — no server logic |
| `tests/` | Unit, integration, contract tests |
| `config/` | Runtime configuration only |
| `docs/` | Architecture, decisions, phases |

Route handlers must stay thin. Business logic goes in services. If you find yourself
adding a database call or complex branching to a route handler, it belongs in a service.

### 4. Tests

- Unit tests for any new logic in `src/inference_x/`
- Unit tests do not require a GPU — mock the engine if needed
- Existing tests must continue to pass: `uv run pytest tests/unit -q`
- If you're fixing a bug, add a test that would have caught it

The current count is 286 passing. A PR that reduces this number will not be merged
unless the removed tests were covering deleted code.

### 5. Record non-obvious decisions

If your PR makes a choice that isn't obvious from the code — a tradeoff, a deliberate
limitation, a rejected alternative — add a `DEC-XXX` entry to `docs/DECISIONS.md`
using the existing template. This is how the project avoids relitigating settled
questions.

### 6. Code style

- Python 3.13+ — use modern type hints (`str | None`, not `Optional[str]`)
- `from __future__ import annotations` at the top of every file
- No third-party formatters are enforced, but match the surrounding code style
- Docstrings on public classes and functions; inline comments only for non-obvious logic
- No `print()` in library code — use the configured logger

---

## Pull request checklist

```
[ ] uv run pytest tests/unit -q — all passing
[ ] No new warnings in pytest output
[ ] File boundaries respected (see AGENTS.md)
[ ] Non-obvious decisions recorded in docs/DECISIONS.md
[ ] PR description explains what changed and why
[ ] If GPU-dependent: tested on WSL2 with a CUDA GPU
```

---

## Reporting bugs

Use GitHub Issues. Include:

1. What you ran (exact command)
2. What you expected
3. What actually happened (full error output)
4. Your environment:
   ```bash
   uv run python --version
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
   uname -r   # kernel version (for WSL2 issues)
   ```

For vLLM-specific failures, also include the relevant lines from
`logs/playground-server.log`.

---

## Questions

Open a GitHub Discussion or file an issue tagged `question`. Response time is
best-effort — this is a one-person project.