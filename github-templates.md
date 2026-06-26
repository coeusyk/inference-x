## Bug report template
## File: .github/ISSUE_TEMPLATE/bug_report.md
---
name: Bug report
about: Something is broken
labels: bug
---

**What happened**
A clear description of the bug.

**To reproduce**
```bash
# exact command
```

**Expected behaviour**
What you expected to happen.

**Actual behaviour**
Full error output or unexpected result.

**Environment**
```
OS: WSL2 Ubuntu 22.04 / Linux / other
Python: (uv run python --version)
GPU: (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
Driver: (nvidia-smi --query-gpu=driver_version --format=csv,noheader)
WSL kernel: (uname -r)
```

**Logs**
If relevant, paste from `logs/playground-server.log` or the terminal output.

---
## Feature request template
## File: .github/ISSUE_TEMPLATE/feature_request.md
---
name: Feature request
about: Propose something new
labels: enhancement
---

**Problem**
What specific problem are you solving? Be concrete.

**Proposed solution**
What would you build? How does it fit the existing architecture?

**Alternatives considered**
What else did you consider and why did you reject it?

**Scope check**
- Does this require authentication or rate limiting? (Out of scope — see DEC-DEFER-01/02)
- Does this change the OpenAI-compatible API contract? (Breaking changes not accepted)
- Does this require a non-vLLM backend? (Not in scope currently)

---
## PR template
## File: .github/PULL_REQUEST_TEMPLATE.md

## What this does
<!-- One or two sentences. -->

## Why
<!-- What problem does it solve? Link the issue if one exists. -->

## Changes
<!-- List the files changed and what each does. -->

## Checklist
- [ ] `uv run pytest tests/unit -q` — all passing
- [ ] No new warnings in pytest output
- [ ] File boundaries respected (see `AGENTS.md`)
- [ ] Non-obvious decisions recorded in `docs/DECISIONS.md`
- [ ] If GPU-dependent: tested on WSL2 with a CUDA GPU