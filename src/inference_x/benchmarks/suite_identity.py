"""Single canonicalizer for benchmark suite identity (DEC-054).

This module is the sole implementation of the `suite_version` content hash. Every
producer and consumer — suite load verification and the `make suite-version`
regeneration command — MUST call it. No other module may reimplement the hash.

Identity is computed over the *parsed in-memory prompt collection*. File encoding,
whitespace, indentation, line endings, JSON object key order, and serialization
formatting are excluded; only prompt ordering and prompt values participate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_SUITE_PATH = "benchmarks/prompts/standard.json"


class SuiteIdentityError(ValueError):
    """Raised when a suite's stored `suite_version` is missing or does not match."""


def compute_suite_version(prompts: list[dict]) -> str:
    """Return the pinned SHA-256 digest of a parsed prompt collection (DEC-054)."""
    canonical = json.dumps(
        prompts, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_suite(data: dict) -> tuple[str, list[dict]]:
    """Return ``(suite_version, prompts)`` after verifying the stored digest.

    Raises :class:`SuiteIdentityError`, naming ``make suite-version``, when the
    ``suite_version`` key is missing or disagrees with the computed digest.
    """
    if "suite_version" not in data:
        raise SuiteIdentityError(
            "Suite is missing the 'suite_version' key. "
            "Generate it with `make suite-version`."
        )
    prompts = data["prompts"]
    stored = data["suite_version"]
    computed = compute_suite_version(prompts)
    if computed != stored:
        raise SuiteIdentityError(
            f"Suite version mismatch: stored {stored}, computed {computed}. "
            "Regenerate it with `make suite-version`."
        )
    return stored, prompts


def suite_version_of(suite_path: str = DEFAULT_SUITE_PATH) -> str:
    """Load and verify a suite file, returning its verified ``suite_version``."""
    data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    return verify_suite(data)[0]
