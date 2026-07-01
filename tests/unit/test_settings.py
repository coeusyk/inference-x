"""Unit tests for AppSettings config loading."""
import os
import textwrap
from pathlib import Path

import pytest
import yaml

from inference_x.core.settings import AppSettings, get_settings


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        textwrap.dedent(
            """\
            models:
              - name: test-model
                engine: vllm
                model_path: facebook/opt-125m
                gpu_memory_utilization: 0.5
              - name: other-model
                engine: vllm
                model_path: some/other-model
            """
        )
    )
    server_yaml = tmp_path / "server.yaml"
    server_yaml.write_text("server:\n  host: 127.0.0.1\n  port: 9000\n")
    return tmp_path


class TestAppSettings:
    def test_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "my-model")
        settings = AppSettings()
        assert settings.default_model == "my-model"

    def test_default_model_fallback(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_X_DEFAULT_MODEL", raising=False)
        settings = AppSettings()
        assert settings.default_model == "qwen2.5-0.5b"

    def test_get_model_config_found(self, tmp_config):
        settings = AppSettings()
        settings.config_dir = str(tmp_config)
        settings.default_model = "test-model"
        cfg = settings.get_model_config()
        assert cfg["model_path"] == "facebook/opt-125m"

    def test_get_model_config_not_found_raises(self, tmp_config):
        settings = AppSettings()
        settings.config_dir = str(tmp_config)
        with pytest.raises(ValueError, match="not found"):
            settings.get_model_config("nonexistent-model")

    def test_get_model_config_missing_file_raises(self, tmp_path):
        settings = AppSettings()
        settings.config_dir = str(tmp_path)
        with pytest.raises(FileNotFoundError):
            settings.get_model_config("anything")

    def test_get_server_config(self, tmp_config):
        settings = AppSettings()
        settings.config_dir = str(tmp_config)
        server = settings.get_server_config()
        assert server["host"] == "127.0.0.1"
        assert server["port"] == 9000

    def test_get_server_config_missing_returns_empty(self, tmp_path):
        settings = AppSettings()
        settings.config_dir = str(tmp_path)
        assert settings.get_server_config() == {}

    def test_loaded_models_defaults_to_default_model(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_X_LOADED_MODELS", raising=False)
        monkeypatch.setenv("INFERENCE_X_DEFAULT_MODEL", "qwen2.5-0.5b")
        settings = AppSettings()
        assert settings.loaded_models == ["qwen2.5-0.5b"]

    def test_loaded_models_from_env(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", "alpha,beta,gamma")
        settings = AppSettings()
        assert settings.loaded_models == ["alpha", "beta", "gamma"]

    def test_loaded_models_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_X_LOADED_MODELS", " alpha , beta ")
        settings = AppSettings()
        assert settings.loaded_models == ["alpha", "beta"]

    def test_get_vram_tier_resolves_from_probed_hardware(self, tmp_config, monkeypatch):
        from inference_x.benchmarks.schemas import HardwareProfile
        from inference_x.utils.vram_tiers import load_tiers

        (tmp_config / "vram_tiers.yaml").write_text(
            textwrap.dedent(
                """\
                tiers:
                  - name: 6gb
                    min_vram_gb: 0
                    description: floor
                    gpu_memory_utilization_ceiling: 0.9
                    max_model_len_cap: 2048
                    max_num_seqs: 4
                    block_size: 16
                    kv_cache_dtype: auto
                  - name: 12gb
                    min_vram_gb: 10
                    description: mid
                    gpu_memory_utilization_ceiling: 0.9
                    max_model_len_cap: 4096
                    max_num_seqs: 8
                    block_size: 16
                    kv_cache_dtype: auto
                """
            )
        )
        load_tiers.cache_clear()
        monkeypatch.setattr(
            "inference_x.benchmarks.hardware.profile_hardware",
            lambda: HardwareProfile(
                gpu_name="Fake GPU",
                vram_total_gb=12.0,
                vram_free_gb=10.0,
                cpu_cores=8,
                ram_total_gb=32.0,
                has_gpu=True,
            ),
        )
        settings = AppSettings()
        settings.config_dir = str(tmp_config)
        tier = settings.get_vram_tier()
        assert tier.name == "12gb"
