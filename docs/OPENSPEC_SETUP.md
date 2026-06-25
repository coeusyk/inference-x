# OpenSpec Setup for InferenceX

## What to install

OpenSpec uses a CLI workflow that is installed globally with npm, then initialized inside the target project.

## Install commands

Run these inside WSL2 so the tooling matches the Linux-based environment used for vLLM work.

```bash
node --version
npm install -g @fission-ai/openspec@latest
openspec --version
```

OpenSpec’s getting-started guidance says Node.js 20.19.0 or higher should be available before installation.

## Initialize in the repo

From the project root, initialize OpenSpec so it creates the `openspec/` directory and assistant-facing instructions.

```bash
cd /path/to/InferenceX
openspec init
```

The documented structure includes `openspec/changes/`, change artifacts such as `proposal.md`, `design.md`, and `tasks.md`, plus durable specs under `openspec/specs/`.

## Suggested workflow for InferenceX

For this project, use OpenSpec for any feature, refactor, or architecture change that spans multiple files or affects contracts.

1. Create a new change with `openspec new change <name>`. 
2. Fill out `proposal.md`, then spec deltas, then `design.md`, then `tasks.md`. 
3. Implement tasks in small batches using Cursor after review.
4. Validate with `openspec validate <name> --json`.
5. Archive with `openspec archive <name> --yes`.

## Recommended first changes

A good sequence for InferenceX is:
- `bootstrap-phase-1`
- `add-core-vllm-engine`
- `add-model-registry-routing`
- `add-observability-pipeline`
- `add-playground-eval`

## Cursor usage notes

OpenSpec has a documented Cursor-oriented workflow and community usage pattern where changes are proposed and applied through structured commands or equivalent manual steps inside the editor.

Even if the exact slash-command integration varies by local setup, the main value is the same: keep planning artifacts in the repo and make Cursor implement against them instead of against vague chat history.