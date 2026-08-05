# OpenSpec Agent Instructions for InferenceX

This repository uses OpenSpec for spec-driven development.

## Working agreement

- Do not implement non-trivial work without an active change under `openspec/changes/`.
- Before making changes, read:
  - `docs/ARCHITECTURE.md`
  - `docs/PHASES.md`
  - `docs/DECISIONS.md`
  - `docs/REVIEW_MODE.md`
  - `docs/BUILD_MODE.md`
  - relevant files in `openspec/specs/`
  - active change artifacts in `openspec/changes/<change-id>/`
- For each change, work in this order:
  1. `proposal.md`
  2. spec deltas under `specs/`
  3. `design.md`
  4. `tasks.md`
  5. implementation
  6. validation
  7. archive

## Agent tooling (mandatory — token efficiency)

Token efficiency is a standing requirement for every session in this repo, not a
per-task preference. Prefer the smallest useful context: filter command output,
sandbox analysis so only answers enter the conversation, and look up symbols
through indexes instead of dumping files.

### Two command paths only — no third

Every shell command must run through **one** of:

| Path | When | Why |
|---|---|---|
| **context-mode** (`ctx_execute`, `ctx_batch_execute`, `ctx_execute_file`, …) | Default for read-only work: gates (`pytest`/`ruff`/`mypy`), `openspec validate`, greps, inspections, scripts that only need stdout | Sandboxes execution so **only stdout/stderr you print** enter the model context (context-mode targets ~98% reduction on tool output). Large results stay indexed; follow up with `ctx_search`. |
| **`rtk proxy <cmd>`** | Host-state mutations that must persist: git commits/branches/merges/pushes, `openspec archive`, moves/renames outside Write/Edit, long-running processes whose exit code on the real host matters | `ctx_execute` runs in a subprocess and **discards filesystem writes**. Mutations that must land in the real worktree leave context-mode via `rtk proxy`. |

A bare `Bash`/`Shell` call, or bare `rtk <cmd>` without `proxy` when you meant a
persisting mutation, is wrong for this repo.

### RTK proxy ([rtk-ai/rtk](https://github.com/rtk-ai/rtk))

RTK is a CLI proxy that filters and compresses command output before the agent
reads it (typically **60–90% fewer bash-output bytes** on supported commands;
git often 85–99%). Docs: Context7 library `/rtk-ai/rtk`, site `https://www.rtk-ai.app`.

**In this repo, prefer:**

```bash
rtk proxy git status
rtk proxy git commit -m "..."
rtk proxy git push -u origin HEAD
rtk proxy openspec archive <change-id>
```

For read-only gates inside context-mode, wrap the same tools so output stays
compact when it does reach stdout, e.g. `rtk pytest …`, `rtk ruff …`, `rtk mypy …`,
`rtk uv run …` — still invoked **via** `ctx_execute` / `ctx_batch_execute`, not as
a bare shell tool call.

Useful analytics: `rtk gain`, `rtk discover`, `rtk session`. Cursor install:
`rtk init -g --agent cursor` (rewrites Bash tool calls when the hook is active).

### context-mode plugin ([mksglu/context-mode](https://github.com/mksglu/context-mode))

Mandatory routing for gather → analyze → recall:

1. **Gather** — `ctx_batch_execute` (multi-command, auto-index) or `ctx_execute`
2. **Web / large docs** — `ctx_fetch_and_index` then `ctx_search` (do not paste full pages)
3. **Project knowledge** — `ctx_index` then `ctx_search` (FTS5 / BM25)
4. **File-heavy analysis** — `ctx_execute_file` (file stays in sandbox; log only the answer)
5. **Health** — `ctx_doctor`, `ctx_stats` (savings ratio this session)

Think-in-code: write a short script that computes the answer; do not Read dozens of
files into the conversation to count or compare by eye. Native `Read` is reserved
for when the next step is an exact `Edit`/`StrReplace` that needs byte-accurate
context.

### token-savior MCP (if present)

When the **token-savior** MCP server is connected in the session, use it **before**
grep-then-read or spawning an explore agent for symbol / impact questions:

- Navigation: `find_symbol`, `get_full_context`, `get_function_source`, `get_class_source`, `search_codebase`, `search_in_symbols`
- Impact: `get_call_chain`, `get_change_impact`, `get_dependents`, `get_dependencies`, `get_edit_context`, `find_impacted_test_files`
- Repo status (structured): `get_git_status`, `get_changed_symbols`, `get_project_summary`

If token-savior is **not** listed in the available MCP servers for this session,
say so once and fall back to context-mode (`ctx_execute` + `rg`) — do not invent
tool calls. Local install may exist (`token-savior` CLI / `.token-savior-cache.json`)
even when the MCP is not attached to Cursor; MCP presence is what matters for
agents.

### Why this stack reduces tokens

| Layer | What it cuts | Mechanism |
|---|---|---|
| RTK | Bash/tool stdout noise | Filters progress bars, passing tests, padding; agent sees failures and summaries |
| context-mode | Bulk tool payloads in the chat | Sandbox keeps raw output; only logged answers + small search hits enter context |
| token-savior | Blind file dumps for navigation | Indexed symbol graph returns location/source/deps without loading whole files |

Together they attack the usual agents tax: huge `pytest`/`git` dumps, repeated
full-file reads, and re-fetching docs already seen this session.

## Branches, commits, and GitHub rulesets

### Long-lived branches

| Branch | Role |
|---|---|
| `develop` | Default integration branch for Phase work and feature PRs |
| `main` | Release / protected trunk |

Do **not** push commits directly to `main` or `develop` for feature work. Cut a
feature branch, open a PR, wait for CI.

### How to start work

1. Sync: `rtk proxy git fetch origin && rtk proxy git checkout develop && rtk proxy git pull --ff-only`
2. Branch from `develop`: `rtk proxy git checkout -b feat/<short-topic>` (or `fix/…`, `chore/…`, `docs/…`)
3. One concern per branch / PR (see `CONTRIBUTING.md`)
4. Commit only when asked (or when the user explicitly requests a commit); use
   conventional, why-focused messages
5. Push: `rtk proxy git push -u origin HEAD`
6. Open PR **into `develop`** (or `main` only for release/promotion PRs)

### Enforced protections (verified on GitHub)

**Ruleset — [Protect Main](https://github.com/coeusyk/inference-x/rules/18108093)**  
Target: default branch (`main` / `~DEFAULT_BRANCH`), enforcement **active**.

- Block branch deletion
- Block force-push / non-fast-forward
- Block creating matching refs outside the allowed flow
- Pull request required before merge; **code owner review** required
- Allowed merge methods: merge, squash, rebase

**Classic branch protection — `develop` and `main`** (OS-1 enforcement):

- Required status check: **`checks`** (job name from `.github/workflows/ci.yml`)
- Both branches report `protected: true` with that required context

A red `checks` run **blocks merge**. Local preflight (same as CI):

```bash
uv run pytest tests/unit -q
uv run ruff check .
uv run mypy src/
```

Prefer running those via context-mode + `rtk` wrappers so output stays compact.

## Repo-specific rules

- InferenceX is a backend-first, layered LLM inference platform.
- Keep the public API stable.
- Prefer additive changes over breaking rewrites only when they genuinely reduce risk.
- Keep route handlers thin and move orchestration to services.
- Use typed schemas and config-driven behavior.
- Develop for WSL2 Ubuntu as the primary local runtime.
- Do not over-index on tiny phases if it slows the project down without reducing risk.
- Prefer one clear vertical slice for a feature rather than excessive pre-scaffolding.

## File boundaries

- `docs/` contains architecture, phase, decision, review, and build docs only.
- `openspec/` contains specs, proposals, designs, tasks, and change deltas only.
- `config/` contains environment and runtime configuration only.
- `src/inference_x/api/` contains FastAPI wiring, routes, dependencies, and error mapping.
- `src/inference_x/services/` contains application orchestration and use-case logic.
- `src/inference_x/engines/` contains engine interfaces and inference implementations.
- `src/inference_x/routing/` contains model selection policies and routing logic.
- `src/inference_x/observability/` contains middleware, metrics, storage, and exporters.
- `src/inference_x/schemas/` contains request/response models only.
- `src/inference_x/benchmarks/` contains hardware profiling, benchmark runner, storage, and advisor logic.
- `benchmarks/prompts/` contains versioned benchmark prompt suites (not Python code).
- `tests/` contains unit, integration, and contract tests only.

## Phase 1 contract

Phase 1 must remain focused on one stable inference path.

Required endpoints (Phase 1):
- `POST /v1/chat/completions`
- `GET /health`

Additional endpoints (later phases, additive only):
- `GET /v1/models` (Phase 2)
- `GET /v1/benchmark/results`, `GET /v1/benchmark/advise` (Phase 6)

Phase 1 expectations:
- one configured vLLM-backed model
- typed request validation
- typed response formatting
- clear error handling for configuration, model load, and inference failures
- smoke-testable local execution in WSL2

## Anti-scope rules

Do not add these before the relevant phase / an authorizing ADR:
- model routing policies beyond a minimal Phase 1 dependency
- observability dashboards or storage backends
- playground UI work
- a second concrete inference backend (requires a future accepted ADR; DEC-047)
- `inference_x/execution/` or other backend-neutral packages/contracts until
  justified by at least two concrete backend implementations (DEC-047)
- optional-extra `vllm` (DEC-007 stands until a second-backend vertical slice)
- config sprawl without validation

## Engine Boundary (DEC-047)

- Backend plurality is a long-term architectural direction; vLLM is the sole
  supported backend today; no second backend is scheduled.
- Allowed hygiene includes:
  - `engines/registry` factory
  - durable capability methods on `BaseEngine` (at minimum `count_prompt_tokens`)
  - typed, observable admission degradation when a capability is unavailable
- Forbidden until justified: thick Execution Contract, speculative backend-neutral
  IR, implementing a second backend without a future ADR.
- Engine Boundary hygiene must not gate Phase B (AsyncLLM).
- Do not knowingly hard-code new vLLM-only assumptions into the composition root,
  `BaseEngine` surface, or admission capability discovery.

## Validation rules

- Every non-trivial change must include tests or a smoke check.
- Contract changes must update specs and docs in the same change.
- Engine changes must be validated against the Phase 1 contract.
- If a change cannot be validated, do not mark it complete.

## Phase discipline

- Follow `docs/PHASES.md`.
- Do not let phase planning block obvious, low-risk progress.
- Record non-obvious design choices in `docs/DECISIONS.md`.
