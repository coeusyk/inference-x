# CLAUDE.md — InferenceX

Repo-local agent instructions for Claude Code / Cursor. Canonical OpenSpec and
architecture rules live in [`AGENTS.md`](AGENTS.md). **Read `AGENTS.md` first**
and follow it for file boundaries, Engine Boundary (DEC-047), OpenSpec order,
and anti-scope.

## Tooling (mandatory every session)

Token efficiency is standing policy. Full rationale and command tables:
[`AGENTS.md` → Agent tooling](AGENTS.md#agent-tooling-mandatory--token-efficiency).

### Two paths only — no third

| Default | Escapes only when |
|---|---|
| **context-mode** — `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` / `ctx_search` / `ctx_fetch_and_index` / `ctx_index` | — |
| **`rtk proxy <cmd>`** | Host mutation that must persist: git, `openspec archive`, real-FS moves, host exit-code-critical runs |

Never use bare `Bash`/`Shell` for gates or inspection. Never use bare `rtk <cmd>`
when the operation must persist — use `rtk proxy`. Prefer wrapping pytest/ruff/mypy/uv
with `rtk …` **inside** context-mode so stdout stays compact.

### token-savior MCP (if present)

If `token-savior` appears in the session MCP catalog, prefer its tools over
grep-then-read for navigation and impact (`find_symbol`, `get_full_context`,
`search_codebase`, `get_call_chain`, `get_change_impact`, `get_edit_context`, …).
If it is absent, fall back to context-mode — do not invent tool calls.

### Why

- **RTK** — cuts most bash-output bytes (often 60–90%; git frequently higher).
- **context-mode** — keeps raw tool output out of the chat (~98% reduction path);
  retrieve with search.
- **token-savior** — indexed symbols/deps instead of loading whole files.

## Branches and commits

See [`AGENTS.md` → Branches, commits, and GitHub rulesets](AGENTS.md#branches-commits-and-github-rulesets).

Summary:

- Branch from **`develop`**: `feat/…`, `fix/…`, `chore/…`, `docs/…`
- PR into **`develop`** (release promotion → `main` separately)
- Never force-push `main` / `develop`; never delete them
- Required CI job name: **`checks`** (blocks merge on red)
- Ruleset **Protect Main** requires PRs + code-owner review on `main`
- Commit only when the user asks; one concern per PR

## OpenSpec

Non-trivial work needs an active change under `openspec/changes/`. Order:
proposal → specs → design → tasks → implementation → validation → archive.
