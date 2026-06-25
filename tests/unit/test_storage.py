"""Unit tests for ResultStore: save/load round-trips and latest_per_model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_x.benchmarks.schemas import BenchmarkResult, HardwareProfile, PromptResult
from inference_x.benchmarks.storage import ResultStore


def _make_result(
    model_name: str = "test-model",
    timestamp: str = "2026-06-08T12:00:00+00:00",
    throughput: float = 30.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        suite_version="abc123",
        timestamp=timestamp,
        concurrency=1,
        prompt_results=[
            PromptResult(
                prompt_label="q1",
                tokens_generated=20,
                ttft_ms=100.0,
                total_latency_ms=800.0,
                tokens_per_sec=throughput,
            )
        ],
        p50_latency_ms=800.0,
        p95_latency_ms=850.0,
        p99_latency_ms=900.0,
        mean_throughput_tps=throughput,
        peak_vram_delta_gb=0.5,
    )


class TestResultStoreRoundTrip:
    def test_save_creates_json_file(self, tmp_path: Path):
        store = ResultStore()
        result = _make_result()
        path = store.save(result, output_dir=str(tmp_path))
        assert path.exists()
        assert path.suffix == ".json"

    def test_saved_json_is_valid(self, tmp_path: Path):
        store = ResultStore()
        result = _make_result()
        path = store.save(result, output_dir=str(tmp_path))
        data = json.loads(path.read_text())
        assert data["model_name"] == "test-model"
        assert data["suite_version"] == "abc123"

    def test_all_results_loads_saved(self, tmp_path: Path):
        store = ResultStore()
        result = _make_result()
        store.save(result, output_dir=str(tmp_path))
        loaded = store.all_results(output_dir=str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].model_name == "test-model"

    def test_all_results_empty_when_dir_missing(self, tmp_path: Path):
        store = ResultStore()
        results = store.all_results(output_dir=str(tmp_path / "nonexistent"))
        assert results == []

    def test_round_trip_preserves_fields(self, tmp_path: Path):
        store = ResultStore()
        original = _make_result(throughput=42.5)
        store.save(original, output_dir=str(tmp_path))
        loaded_list = store.all_results(output_dir=str(tmp_path))
        assert len(loaded_list) == 1
        loaded = loaded_list[0]
        assert loaded.mean_throughput_tps == 42.5
        assert loaded.p50_latency_ms == 800.0
        assert len(loaded.prompt_results) == 1
        assert loaded.prompt_results[0].prompt_label == "q1"

    def test_result_serialises_hardware(self, tmp_path: Path, make_hardware):
        store = ResultStore()
        hw = make_hardware()
        original = _make_result()
        original.hardware = hw
        store.save(original, output_dir=str(tmp_path))
        loaded = store.all_results(output_dir=str(tmp_path))[0]
        assert loaded.hardware is not None
        assert loaded.hardware.gpu_name == hw.gpu_name
        assert loaded.hardware.vram_total_gb == hw.vram_total_gb

    def test_result_without_hardware_deserialises(self, tmp_path: Path):
        store = ResultStore()
        legacy = {
            "model_name": "legacy-model",
            "suite_version": "abc123",
            "timestamp": "2026-06-08T12:00:00+00:00",
            "concurrency": 1,
            "prompt_results": [],
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_throughput_tps": 10.0,
            "peak_vram_delta_gb": 1.0,
        }
        path = tmp_path / "results-legacy-model-2026-06-08T12-00-00.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = store.all_results(output_dir=str(tmp_path))[0]
        assert loaded.hardware is None
        assert loaded.model_name == "legacy-model"

    def test_output_dir_created_if_absent(self, tmp_path: Path):
        store = ResultStore()
        nested = tmp_path / "a" / "b" / "c"
        result = _make_result()
        path = store.save(result, output_dir=str(nested))
        assert path.exists()


class TestLatestPerModel:
    def test_returns_most_recent_per_model(self, tmp_path: Path):
        store = ResultStore()
        older = _make_result(timestamp="2026-06-07T10:00:00+00:00", throughput=20.0)
        newer = _make_result(timestamp="2026-06-08T10:00:00+00:00", throughput=35.0)
        store.save(older, output_dir=str(tmp_path))
        store.save(newer, output_dir=str(tmp_path))
        latest = store.latest_per_model(output_dir=str(tmp_path))
        assert "test-model" in latest
        assert latest["test-model"].mean_throughput_tps == 35.0

    def test_multiple_models(self, tmp_path: Path):
        store = ResultStore()
        r_a = _make_result(model_name="model-a", throughput=10.0)
        r_b = _make_result(model_name="model-b", throughput=50.0)
        store.save(r_a, output_dir=str(tmp_path))
        store.save(r_b, output_dir=str(tmp_path))
        latest = store.latest_per_model(output_dir=str(tmp_path))
        assert "model-a" in latest
        assert "model-b" in latest
        assert latest["model-a"].mean_throughput_tps == 10.0
        assert latest["model-b"].mean_throughput_tps == 50.0

    def test_empty_when_no_results(self, tmp_path: Path):
        store = ResultStore()
        latest = store.latest_per_model(output_dir=str(tmp_path / "empty"))
        assert latest == {}
