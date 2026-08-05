"""Unit tests for the shared suite-identity canonicalizer (DEC-054)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_x.benchmarks import runner
from inference_x.benchmarks.suite_identity import (
    SuiteIdentityError,
    compute_suite_version,
    suite_version_of,
    verify_suite,
)

PINNED_DIGEST = "b47066414716cf4a0970adc790384f5173bd488cf80da47d28936b3d2ce5cfa4"
STANDARD_SUITE = Path(__file__).parents[2] / "benchmarks" / "prompts" / "standard.json"


def _standard_prompts() -> list[dict]:
    return json.loads(STANDARD_SUITE.read_text(encoding="utf-8"))["prompts"]


class TestCharacterization:
    def test_shipped_suite_matches_pinned_digest(self):
        assert compute_suite_version(_standard_prompts()) == PINNED_DIGEST

    def test_suite_version_of_shipped_file(self):
        assert suite_version_of(str(STANDARD_SUITE)) == PINNED_DIGEST


class TestIdentityExclusions:
    def test_object_key_order_excluded(self):
        prompts = [{"label": "a", "text": "hello"}]
        reordered = [{"text": "hello", "label": "a"}]
        assert compute_suite_version(prompts) == compute_suite_version(reordered)

    def test_whitespace_and_formatting_excluded(self):
        """A round-trip through pretty-printed JSON does not change identity."""
        prompts = _standard_prompts()
        pretty = json.loads(json.dumps(prompts, indent=4))
        assert compute_suite_version(pretty) == compute_suite_version(prompts)


class TestIdentitySensitivity:
    def test_text_change_changes_digest(self):
        base = [{"label": "a", "text": "hello"}]
        changed = [{"label": "a", "text": "hello!"}]
        assert compute_suite_version(base) != compute_suite_version(changed)

    def test_label_change_changes_digest(self):
        base = [{"label": "a", "text": "hello"}]
        changed = [{"label": "b", "text": "hello"}]
        assert compute_suite_version(base) != compute_suite_version(changed)

    def test_order_change_changes_digest(self):
        base = [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}]
        reordered = [{"label": "b", "text": "y"}, {"label": "a", "text": "x"}]
        assert compute_suite_version(base) != compute_suite_version(reordered)


class TestSuiteVersionKeyExcluded:
    def test_suite_version_field_not_hashed(self):
        prompts = [{"label": "a", "text": "hello"}]
        digest = compute_suite_version(prompts)
        data = {"suite_version": digest, "prompts": prompts}
        # A different suite_version value with identical prompts still verifies as
        # the same computed digest — the key is excluded from the hashed content.
        assert compute_suite_version(data["prompts"]) == digest


class TestVerifySuite:
    def test_matching_suite_returns_version_and_prompts(self):
        prompts = [{"label": "a", "text": "hello"}]
        data = {"suite_version": compute_suite_version(prompts), "prompts": prompts}
        version, returned = verify_suite(data)
        assert version == data["suite_version"]
        assert returned == prompts

    def test_mismatch_raises_naming_command(self):
        data = {"suite_version": "deadbeef", "prompts": [{"label": "a", "text": "x"}]}
        with pytest.raises(SuiteIdentityError, match="make suite-version"):
            verify_suite(data)

    def test_missing_key_raises_actionable_error(self):
        data = {"prompts": [{"label": "a", "text": "x"}]}
        with pytest.raises(SuiteIdentityError, match="make suite-version"):
            verify_suite(data)


class TestSharedImplementation:
    def test_runner_load_suite_uses_shared_verifier(self):
        # The runner must not carry a second hashing path; it imports the shared
        # verifier and _load_suite delegates to it.
        assert runner.verify_suite is verify_suite

    def test_runner_load_suite_rejects_tampered_suite(self, tmp_path: Path):
        bad = tmp_path / "suite.json"
        bad.write_text(
            json.dumps({"suite_version": "nope", "prompts": [{"label": "a", "text": "x"}]}),
            encoding="utf-8",
        )
        with pytest.raises(SuiteIdentityError, match="make suite-version"):
            runner._load_suite(str(bad))
