"""Unit tests for benchmark API routes (GET /v1/benchmark/results and /advise)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from inference_x.api.main import app
from inference_x.benchmarks.schemas import (
    AdvisorReport,
    AdvisorResult,
    BenchmarkResult,
    HardwareProfile,
    PromptResult,
)


def _hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_gb=8.0,
        vram_free_gb=6.0,
        cpu_cores=8,
        ram_total_gb=32.0,
        has_gpu=True,
    )


def _result(model_name: str = "test-model") -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        suite_version="abc123",
        timestamp="2026-06-08T12:00:00+00:00",
        concurrency=1,
        prompt_results=[
            PromptResult(
                prompt_label="q1",
                tokens_generated=20,
                ttft_ms=100.0,
                total_latency_ms=800.0,
                tokens_per_sec=25.0,
            )
        ],
        p50_latency_ms=800.0,
        p95_latency_ms=850.0,
        p99_latency_ms=900.0,
        mean_throughput_tps=25.0,
        peak_vram_delta_gb=1.5,
    )


def _advisor_result(model_name: str = "test-model") -> AdvisorResult:
    return AdvisorResult(
        model_name=model_name,
        score=75.0,
        viable=True,
        throughput_tps=25.0,
        ttft_ms=100.0,
        vram_gb=1.5,
        recommendation_str=f"{model_name} — 25 tok/s, 100ms TTFT, 1.5GB VRAM",
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestBenchmarkResultsRoute:
    def test_returns_200_with_empty_results(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
        ):
            mock_store.all_results.return_value = []
            mock_hw.return_value = _hw()
            resp = client.get("/v1/benchmark/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []
        assert "hardware" in body

    def test_returns_results_list(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
        ):
            mock_store.all_results.return_value = [_result("model-a"), _result("model-b")]
            mock_hw.return_value = _hw()
            resp = client.get("/v1/benchmark/results")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2
        model_names = {r["model_name"] for r in body["results"]}
        assert model_names == {"model-a", "model-b"}

    def test_hardware_profile_in_response(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
        ):
            mock_store.all_results.return_value = []
            mock_hw.return_value = _hw()
            resp = client.get("/v1/benchmark/results")
        body = resp.json()
        hw = body["hardware"]
        assert hw["gpu_name"] == "Test GPU"
        assert hw["vram_free_gb"] == 6.0
        assert hw["has_gpu"] is True


class TestBenchmarkAdviseRoute:
    def test_returns_200_with_no_results(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
            patch("inference_x.api.routes.benchmark._advisor") as mock_advisor,
        ):
            mock_store.latest_per_model.return_value = {}
            mock_hw.return_value = _hw()
            mock_advisor.rank.return_value = AdvisorReport(ranked=[], warnings=[])
            resp = client.get("/v1/benchmark/advise")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ranked"] == []
        assert "hardware" in body
        assert "generated_at" in body

    def test_returns_ranked_results(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
            patch("inference_x.api.routes.benchmark._advisor") as mock_advisor,
        ):
            mock_store.latest_per_model.return_value = {"test-model": _result()}
            mock_hw.return_value = _hw()
            mock_advisor.rank.return_value = AdvisorReport(
                ranked=[_advisor_result()],
                warnings=[],
            )
            resp = client.get("/v1/benchmark/advise")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["ranked"]) == 1
        assert body["ranked"][0]["model_name"] == "test-model"
        assert body["ranked"][0]["viable"] is True

    def test_returns_warnings_when_advisor_reports_mismatch(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
            patch("inference_x.api.routes.benchmark._advisor") as mock_advisor,
        ):
            mock_store.latest_per_model.return_value = {"other-gpu": _result("other-gpu")}
            mock_hw.return_value = _hw()
            mock_advisor.rank.return_value = AdvisorReport(
                ranked=[],
                warnings=["Skipped other-gpu — benchmark was run on RTX 4090 (24.0 GB)"],
            )
            resp = client.get("/v1/benchmark/advise")
        assert resp.status_code == 200
        body = resp.json()
        assert body["warnings"]
        assert "Skipped other-gpu" in body["warnings"][0]

    def test_generated_at_is_iso_string(self, client: TestClient):
        with (
            patch("inference_x.api.routes.benchmark._store") as mock_store,
            patch("inference_x.api.routes.benchmark.profile_hardware") as mock_hw,
            patch("inference_x.api.routes.benchmark._advisor") as mock_advisor,
        ):
            mock_store.latest_per_model.return_value = {}
            mock_hw.return_value = _hw()
            mock_advisor.rank.return_value = AdvisorReport(ranked=[], warnings=[])
            resp = client.get("/v1/benchmark/advise")
        body = resp.json()
        assert isinstance(body["generated_at"], str)
        assert "T" in body["generated_at"]
