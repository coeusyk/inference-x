"""Unit tests for run-manifest content-addressing (Phase C, C1 — add-run-manifest)."""
from __future__ import annotations

import os

from inference_x.schemas.chat import ChatMessage
from inference_x.utils.ids import (
    compute_prompt_sha256,
    compute_run_id,
    is_hf_hub_repo_id,
    package_version,
    resolve_git_sha,
)


class TestComputeRunId:
    def test_identical_preimages_yield_identical_run_id(self):
        a = compute_run_id({"engine": {"name": "x"}, "model": {"registry_name": "m"}})
        b = compute_run_id({"engine": {"name": "x"}, "model": {"registry_name": "m"}})
        assert a == b

    def test_key_order_does_not_affect_run_id(self):
        a = compute_run_id({"a": 1, "b": 2})
        b = compute_run_id({"b": 2, "a": 1})
        assert a == b

    def test_differing_preimage_changes_run_id(self):
        a = compute_run_id({"seed": 1})
        b = compute_run_id({"seed": 2})
        assert a != b

    def test_run_id_has_sha256_prefix_and_hex_digest(self):
        run_id = compute_run_id({"a": 1})
        assert run_id.startswith("sha256:")
        digest = run_id.removeprefix("sha256:")
        assert len(digest) == 64
        int(digest, 16)  # raises if not valid hex


class TestComputePromptSha256:
    def test_identical_messages_yield_identical_hash(self):
        msgs = [ChatMessage(role="user", content="hi")]
        assert compute_prompt_sha256(msgs) == compute_prompt_sha256(msgs)

    def test_different_content_yields_different_hash(self):
        a = [ChatMessage(role="user", content="hi")]
        b = [ChatMessage(role="user", content="bye")]
        assert compute_prompt_sha256(a) != compute_prompt_sha256(b)

    def test_is_a_hex_sha256_digest_without_prefix(self):
        digest = compute_prompt_sha256([ChatMessage(role="user", content="hi")])
        assert len(digest) == 64
        int(digest, 16)


class TestIsHfHubRepoId:
    def test_hub_repo_id_is_recognized(self):
        assert is_hf_hub_repo_id("meta-llama/Llama-3-8B") is True

    def test_absolute_unix_path_is_not_a_hub_id(self):
        assert is_hf_hub_repo_id("/models/llama") is False

    def test_relative_path_is_not_a_hub_id(self):
        assert is_hf_hub_repo_id("./models/llama") is False

    def test_windows_drive_path_is_not_a_hub_id(self):
        assert is_hf_hub_repo_id("C:/models/llama") is False

    def test_empty_string_is_not_a_hub_id(self):
        assert is_hf_hub_repo_id("") is False

    def test_single_segment_is_not_a_hub_id(self):
        assert is_hf_hub_repo_id("llama") is False


class TestPackageVersion:
    def test_installed_package_returns_a_version_string(self):
        assert package_version("inferencex") is not None

    def test_unknown_package_returns_none_never_fabricated(self):
        assert package_version("this-package-does-not-exist-anywhere") is None


class TestResolveGitSha:
    def test_env_override_takes_precedence(self, monkeypatch):
        import inference_x.utils.ids as ids_mod

        monkeypatch.setattr(ids_mod, "_git_sha_cache", ids_mod._UNSET)
        monkeypatch.setenv("INFERENCE_X_GIT_SHA", "deadbeef")
        assert resolve_git_sha() == "deadbeef"
        monkeypatch.setattr(ids_mod, "_git_sha_cache", ids_mod._UNSET)

    def test_result_is_cached_across_calls(self, monkeypatch):
        import inference_x.utils.ids as ids_mod

        monkeypatch.setattr(ids_mod, "_git_sha_cache", ids_mod._UNSET)
        monkeypatch.setenv("INFERENCE_X_GIT_SHA", "cafef00d")
        first = resolve_git_sha()
        os.environ.pop("INFERENCE_X_GIT_SHA", None)
        second = resolve_git_sha()
        assert first == second == "cafef00d"
        monkeypatch.setattr(ids_mod, "_git_sha_cache", ids_mod._UNSET)
