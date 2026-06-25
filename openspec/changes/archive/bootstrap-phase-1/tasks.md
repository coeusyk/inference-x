# Tasks: bootstrap-phase-1

## 1. OpenSpec foundation
- [ ] 1.1 Add `AGENTS.md` to the repository root
- [ ] 1.2 Add `openspec/config.yaml`
- [ ] 1.3 Add `openspec/specs/platform/spec.md`
- [ ] 1.4 Review OpenSpec artifacts against `docs/ARCHITECTURE.md` and `docs/PHASES.md`

## 2. Validate planning model
- [ ] 2.1 Confirm `schema: spec-driven` is appropriate for this repo
- [ ] 2.2 Keep proposal, spec delta, design, and tasks focused on one change only
- [ ] 2.3 Decide whether future low-risk changes should use a lighter custom schema

## 3. Prepare first execution change
- [ ] 3.1 Create a new change named `add-core-vllm-engine`
- [ ] 3.2 Write its proposal
- [ ] 3.3 Add its spec delta for Phase 1 API and engine behavior
- [ ] 3.4 Write its design document
- [ ] 3.5 Break implementation into thin vertical-slice tasks

## 4. Working agreement
- [ ] 4.1 Use OpenSpec for all non-trivial features and refactors
- [ ] 4.2 Update docs and tests in the same work batch when contracts change
- [ ] 4.3 Archive completed changes after validation