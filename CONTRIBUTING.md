# Contributing to InferenceX

InferenceX is a self-hosted LLM inference platform built incrementally. Contributions
are welcome — bug reports, fixes, documentation improvements, and new features that
fit the project's scope and phase roadmap.

***

## Before you start

Read `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` first. The architecture doc explains
the layer boundaries (routes → service → engine → vLLM adapter). The decisions doc
explains *why* things are the way they are — many choices that look unconventional have
a recorded rationale.

If you're planning a non-trivial change, open an issue first and describe what you're
building and why. This avoids duplicate work and ensures the change aligns with the
current phase.

***

## Development setup

**Requirements:** Python 3.13+, [`uv`](https://docs.astral.sh/uv/), a CUDA-capable GPU,
and WSL2 or Linux.

```bash
git clone https://github.com/coeusyk/inference-x
cd inference-x
uv sync
cp .env.example .env          # add HF_TOKEN if using gated models
./scripts/dev.sh serve        # start the dev server
```

For hardware-profiling accuracy on WSL2:

```bash
uv sync --extra hardware      # installs nvidia-ml-py + psutil
```

***

## Project structure

```
src/inference_x/
├── api/            # FastAPI routes — no business logic here
├── core/           # Service layer, engine interface, model registry
├── engine/         # vLLM adapter and engine pool
├── benchmarks/     # Runner, advisor, hardware profiler, storage
├── observability/  # Metrics middleware and ring buffer
└── playground/     # Textual TUI (chat, compare, loading screen)

scripts/            # CLI entry points (advise.py, benchmark.py, etc.)
config/             # models.yaml, logging.yaml
docs/               # ARCHITECTURE.md, DECISIONS.md, PHASES.md
tests/unit/         # All tests — mirrors src/ structure
```

The key constraint: **route handlers must not contain business logic**. Routes call
services; services call engines. If you find yourself writing conditional logic inside
a route handler, it belongs in the service layer.

***

## Making changes

### Branching

Branch from `develop`, not `main`. Use a descriptive name:

```
feat/cold-start-margin
fix/advisor-vram-gate
docs/dec-034-rationale
```

### Code style

The project uses `ruff` for linting and formatting.

```bash
uv run ruff check .
uv run ruff format .
```

Both must pass before opening a pull request.

### Tests

All tests live in `tests/unit/`. Run the full suite with:

```bash
make test
```

**Every code change needs a test.** The bar:
- New functions → at least one happy-path and one failure-mode test
- Bug fixes → a regression test that fails on the old code and passes on the fix
- New edge cases → assert the exact output string or value, not just `is not None`

The current test count is tracked in the README badge. Update it if your PR changes
the count.

### Commit messages

Use the conventional commit format:

```
feat: add cold-start margin to VRAM viability gate
fix: peak_vram_delta_gb always 0 when model pre-loaded
docs: add DEC-034 rationale for cold-start multiplier
test: regression for marginal footprint not viable with margin
refactor: extract _format_vram_requirement helper
```

Reference the issue number in the commit body or use `Fixes #N` to auto-close:

```
feat: implement cold-start margin for advisor viability gate

Fixes #3
```

***

## Pull requests

- Target `develop`
- Keep PRs focused — one logical change per PR
- Include a short description of *what* changed and *why*, not just *how*
- Update `docs/DECISIONS.md` with a `DEC-NNN` entry for any non-obvious design choice
- Update `docs/PHASES.md` if the change relates to a phase deliverable
- The README test badge (`tests-NNN passing`) should reflect the new count

PR titles should follow the same conventional commit format as commit messages.

***

## What to contribute

### Good fits
- Bug fixes with a clear reproduction case
- New benchmark prompt variants (add to `benchmarks/prompts/`)
- Additional hardware profiling accuracy improvements
- Documentation and decision record improvements
- New models added to `config/models.yaml` with tested `gpu_memory_utilization` values
- Observability improvements (new metrics, better aggregation)

### Out of scope for now
- Authentication / rate limiting (deferred — DEC-DEFER-01, DEC-DEFER-02)
- Multi-GPU support
- Docker / container packaging
- Any change that requires modifying the vLLM engine internals directly

If you're unsure whether something fits, open an issue and ask before building.

***

## Reporting bugs

Open an issue with:
- What you ran (`make chat`, `make advise`, etc.)
- What you expected to happen
- What actually happened (include the full error output or log excerpt)
- Your hardware (`nvidia-smi` output or equivalent) and OS

For VRAM or model loading issues, include the output of `make advise` — it shows
your hardware profile and which models are marked viable.

***

## License

By contributing, you agree that your changes will be licensed under the
[MIT License](./LICENSE) that covers this project.
