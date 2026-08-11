"""Run manifest content-addressing (Phase C, C1 — add-run-manifest).

`compute_run_id` is the sole hasher for run identity: it reuses the
canonicalization discipline of `benchmarks/suite_identity.py` (DEC-054) —
`json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ":"))`
then SHA-256 — applied to the run manifest's semantics-determining preimage
rather than a prompt collection. Content-addressing only: no cryptographic
signature is produced (add-run-manifest design.md D1; signing is a Phase E
candidate).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from inference_x.schemas.chat import ChatMessage

_RUN_ID_PREFIX = "sha256:"
_UNSET = object()
_git_sha_cache: object = _UNSET


def compute_run_id(preimage: dict[str, Any]) -> str:
    """Return the content-addressed `run_id` for a manifest preimage.

    *preimage* MUST contain only the run manifest's configuration-identity
    fields — add-run-manifest design.md D2 defines the preimage as
    `{engine, model, runtime, sampling, request, warnings}`, where
    `warnings` carries only each warning's `type`/`code`/`field`. `timing`,
    `batch`, and `hardware` are excluded entirely (speed/observational, and
    execution-environment rather than configuration identity), and a
    warning's free-text `message` is excluded from the `warnings` entries.
    Two preimages that are equal (key order does not matter — keys are
    sorted) produce the same `run_id`; any other difference produces a
    different one.
    """
    canonical = json.dumps(
        preimage, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_RUN_ID_PREFIX}{digest}"


def compute_prompt_sha256(messages: list[ChatMessage]) -> str:
    """Return the content hash of a request's messages (manifest `request.prompt_sha256`).

    Hashes the structured `{role, content}` messages rather than an
    engine-rendered prompt string, so this never crosses the Engine Boundary
    (DEC-047) — combined with `chat_template_sha256`, "same messages + same
    template" is recoverable without the server needing to expose the
    rendered string itself.
    """
    canonical = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_hf_hub_repo_id(model_path: str) -> bool:
    """True when *model_path* looks like a HuggingFace Hub repo id (org/name).

    Deliberately a standalone copy of the classification in
    `engines/vllm_engine.py::_is_hf_hub_model_path` rather than an import of
    it: that function is private to the concrete vLLM backend, and importing
    it here would make `services/` depend on a concrete engine module, which
    the Engine Boundary (DEC-047) forbids. The two must stay behaviourally
    identical; there is no cross-backend contract at stake since this is a
    manifest-provenance classification, not admission or generation logic.
    """
    if not model_path:
        return False
    if model_path.startswith(("/", "./", "../", "~")):
        return False
    if os.path.isabs(model_path):
        return False
    if len(model_path) >= 2 and model_path[1] == ":":
        return False
    parts = model_path.split("/")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


def package_version(name: str) -> str | None:
    """Installed distribution version for *name*, or `None` when not found.

    Never fabricated (add-run-manifest design.md D4): an uninstalled or
    unresolvable distribution reports `None`, not a guessed value.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def resolve_git_sha() -> str | None:
    """Short git commit SHA for this checkout, or `None` (never fabricated).

    `INFERENCE_X_GIT_SHA` takes precedence when set — for build/release
    pipelines packaging a wheel with no `.git` directory. Falls back to
    `git rev-parse --short HEAD` for source checkouts, run from anywhere
    inside the repo (git walks up to find `.git`, so the exact depth of
    this file under the repo root does not matter). Cached for the process
    lifetime: the answer cannot change while the process is running.
    """
    global _git_sha_cache
    if _git_sha_cache is not _UNSET:
        return _git_sha_cache  # type: ignore[return-value]

    env_sha = os.environ.get("INFERENCE_X_GIT_SHA", "").strip()
    if env_sha:
        _git_sha_cache = env_sha
        return env_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent,
        )
        sha = result.stdout.strip()
        _git_sha_cache = sha or None
    except Exception:
        _git_sha_cache = None
    return _git_sha_cache  # type: ignore[return-value]
