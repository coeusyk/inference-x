---
name: cut-release
description: Decide whether unreleased develop work warrants a release, pick the version per semver.org, and cut it following this repo's established commit/PR/tag/release pattern.
---

# Cut a release

Use when asked to release, version-bump, or judge whether accumulated
`develop` work is "worth" a release.

## 1. Decide the version — semver.org rules

Given `pyproject.toml`'s current `version` and the set of OpenSpec changes
merged to `develop` since the last release tag:

- **MAJOR** (`X.0.0`) — any backward-incompatible change to a public
  contract: a route removed/renamed, a response schema field removed or
  repurposed, a behavior change existing callers relied on.
- **MINOR** (`X.Y.0`) — new backward-compatible functionality: a new route,
  a new optional field, a new capability that doesn't touch existing
  contracts.
- **PATCH** (`X.Y.Z`) — backward-compatible bug fixes only, no new surface.

Bump once for the whole batch, not once per change — take the highest tier
any single change in the batch requires. A batch containing one MINOR change
and three PATCH changes is still MINOR.

This repo treats `0.x` releases as real, numbered, user-facing releases
(see existing tags) — don't withhold a release just because major version is
still `0`.

## 2. Confirm release-worthiness

Not every merge needs a release. Cut one when `develop` has diverged from
the last release tag with at least one user-visible change (new endpoint,
behavior fix, dependency-facing change). Pure internal refactors with zero
external surface change (like an engine-driver swap with no API change) are
still worth bundling into the next release rather than triggering one alone
— check `git log <last-tag>..develop --oneline` for what's accumulated.

## 3. This repo's release pattern (mirror it, don't reinvent)

Verified from git history (`git show --stat` on prior release-prep commits,
e.g. `5b81605`, `51c764f`):

1. Feature work already landed on `develop` via normal `feat/`→`develop` PRs
   (each with its own `feat(...)` commit + separate
   `chore(openspec): archive <id>` commit).
2. Branch `chore/release-vX.Y.Z` from `develop`.
3. Bump **only** `pyproject.toml`'s `version` field and regenerate
   `uv.lock` (`uv lock`) — nothing else. `CHANGELOG.md` is NOT touched by
   this commit; there is no repo precedent for closing out
   `## [Unreleased]` into a versioned section, and none should be invented
   without asking.
4. Commit as `chore(release): prepare Phase A vX.Y.Z` (adjust the phase
   name if the repo has moved past Phase A).
5. PR `chore/release-vX.Y.Z` → `develop`, wait for the required `checks`
   CI job, merge.
6. PR `develop` → `main` (title: `Release vX.Y.Z: <short theme>`), wait for
   `checks`, merge. `main` requires code-owner review per the "Protect
   Main" ruleset — if merge is blocked on review, stop and ask rather than
   overriding it.
7. On `main`, tag: `git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
8. `gh release create vX.Y.Z --title "vX.Y.Z — <theme>" --notes-file ...`
   with a hand-authored narrative body (NOT a copy of `CHANGELOG.md`'s
   terse bullets) structured as:
   - Title: `InferenceX vX.Y.Z — <thematic summary>`
   - `### What changed` — one subsection per OpenSpec change, each a
     prose paragraph naming the change id and explaining what changed and
     why, in the style of prior releases (`gh release view v0.3.0 --json
     body` for reference).
   - Optional `### Also` catch-all for minor/doc-only additions.
   - `**Full Changelog**: https://github.com/<owner>/<repo>/compare/vPREV...vNEW`

## 4. Tooling

Every git/gh command in this flow goes through `rtk proxy` (git commits,
branches, tags, PR create/merge, release create). Read-only file inspection
goes through context-mode. See `AGENTS.md` → Agent tooling.
