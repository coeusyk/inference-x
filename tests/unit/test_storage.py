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
    suite_version: str = "abc123",
) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        suite_version=suite_version,
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
        vram_device_occupied_gib=0.5,
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


class TestSuiteAwareSelection:
    def test_filters_before_latest_per_model(self, tmp_path: Path):
        store = ResultStore()
        # Newer result is a different suite; older matches the expected suite.
        matching = _make_result(
            timestamp="2026-06-07T10:00:00+00:00", throughput=20.0, suite_version="v1"
        )
        newer_other = _make_result(
            timestamp="2026-06-08T10:00:00+00:00", throughput=99.0, suite_version="v2"
        )
        store.save(matching, output_dir=str(tmp_path))
        store.save(newer_other, output_dir=str(tmp_path))

        selection = store.latest_per_model_for_suite("v1", output_dir=str(tmp_path))
        assert selection.status == "ok"
        assert set(selection.latest) == {"test-model"}
        # The newer, non-matching suite result did not win — filtering happened
        # before latest-per-model.
        assert selection.latest["test-model"].mean_throughput_tps == 20.0

    def test_latest_among_matching_suite(self, tmp_path: Path):
        store = ResultStore()
        older = _make_result(
            timestamp="2026-06-07T10:00:00+00:00", throughput=20.0, suite_version="v1"
        )
        newer = _make_result(
            timestamp="2026-06-08T10:00:00+00:00", throughput=35.0, suite_version="v1"
        )
        store.save(older, output_dir=str(tmp_path))
        store.save(newer, output_dir=str(tmp_path))
        selection = store.latest_per_model_for_suite("v1", output_dir=str(tmp_path))
        assert selection.status == "ok"
        assert selection.latest["test-model"].mean_throughput_tps == 35.0

    def test_same_suite_multiple_models_multiple_timestamps(self, tmp_path: Path):
        """Within one suite, newest result per model is still selected."""
        store = ResultStore()
        store.save(
            _make_result(
                model_name="model-a",
                timestamp="2026-06-07T10:00:00+00:00",
                throughput=10.0,
                suite_version="v1",
            ),
            output_dir=str(tmp_path),
        )
        store.save(
            _make_result(
                model_name="model-a",
                timestamp="2026-06-08T10:00:00+00:00",
                throughput=22.0,
                suite_version="v1",
            ),
            output_dir=str(tmp_path),
        )
        store.save(
            _make_result(
                model_name="model-b",
                timestamp="2026-06-07T11:00:00+00:00",
                throughput=40.0,
                suite_version="v1",
            ),
            output_dir=str(tmp_path),
        )
        store.save(
            _make_result(
                model_name="model-b",
                timestamp="2026-06-08T11:00:00+00:00",
                throughput=55.0,
                suite_version="v1",
            ),
            output_dir=str(tmp_path),
        )
        selection = store.latest_per_model_for_suite("v1", output_dir=str(tmp_path))
        assert selection.status == "ok"
        assert set(selection.latest) == {"model-a", "model-b"}
        assert selection.latest["model-a"].mean_throughput_tps == 22.0
        assert selection.latest["model-b"].mean_throughput_tps == 55.0

    def test_empty_store_status(self, tmp_path: Path):
        store = ResultStore()
        selection = store.latest_per_model_for_suite(
            "v1", output_dir=str(tmp_path / "empty")
        )
        assert selection.status == "empty"
        assert selection.latest == {}

    def test_suite_mismatch_status_distinct_from_empty(self, tmp_path: Path):
        store = ResultStore()
        store.save(
            _make_result(suite_version="v2"), output_dir=str(tmp_path)
        )
        selection = store.latest_per_model_for_suite("v1", output_dir=str(tmp_path))
        assert selection.status == "suite_mismatch"
        assert selection.latest == {}

    def test_mismatch_does_not_delete_files(self, tmp_path: Path):
        store = ResultStore()
        store.save(_make_result(suite_version="v2"), output_dir=str(tmp_path))
        before = list(tmp_path.glob("results-*.json"))
        store.latest_per_model_for_suite("v1", output_dir=str(tmp_path))
        after = list(tmp_path.glob("results-*.json"))
        assert before == after and len(after) == 1


class TestHistoricalCorpusLoads:
    """DEC-057: the 25 historical result files are the real compatibility corpus."""

    _CORPUS = Path("benchmarks/results")

    def test_all_historical_results_load(self):
        store = ResultStore()
        files = sorted(self._CORPUS.glob("*.json"))
        if not files:
            pytest.skip("no historical benchmark corpus present")
        loaded = store.all_results(output_dir=str(self._CORPUS))
        assert len(loaded) == len(files)
        # Every file exposes the canonical VRAM view (alias-populated when the
        # file predates the rename).
        for r in loaded:
            assert isinstance(r.vram_device_occupied_gib, float)


class TestVramAliasRoundTrip:
    """DEC-057 canonical/alias precedence (V13a–c) exercised through the schema."""

    def test_canonical_only_round_trip(self, tmp_path: Path):
        store = ResultStore()
        original = _make_result()  # writes canonical via helper
        store.save(original, output_dir=str(tmp_path))
        raw = json.loads(next(tmp_path.glob("results-*.json")).read_text())
        assert "vram_device_occupied_gib" in raw
        assert "peak_vram_delta_gb" not in raw
        loaded = store.all_results(output_dir=str(tmp_path))[0]
        assert loaded.vram_device_occupied_gib == 0.5

    def test_alias_only_loads(self):
        loaded = BenchmarkResult.model_validate(
            {
                "model_name": "legacy",
                "suite_version": "v1",
                "timestamp": "2026-06-08T12:00:00+00:00",
                "mean_throughput_tps": 10.0,
                "peak_vram_delta_gb": 1.25,
            }
        )
        assert loaded.vram_device_occupied_gib == 1.25

    def test_equal_dual_values_load(self):
        loaded = BenchmarkResult.model_validate(
            {
                "model_name": "dual",
                "suite_version": "v1",
                "timestamp": "2026-06-08T12:00:00+00:00",
                "mean_throughput_tps": 10.0,
                "vram_device_occupied_gib": 3.0,
                "peak_vram_delta_gb": 3.0,
            }
        )
        assert loaded.vram_device_occupied_gib == 3.0

    def test_conflicting_dual_values_canonical_wins(self):
        loaded = BenchmarkResult.model_validate(
            {
                "model_name": "conflict",
                "suite_version": "v1",
                "timestamp": "2026-06-08T12:00:00+00:00",
                "mean_throughput_tps": 10.0,
                "vram_device_occupied_gib": 4.0,
                "peak_vram_delta_gb": 9.9,
            }
        )
        assert loaded.vram_device_occupied_gib == 4.0
