"""Unit tests for ModelAdvisor scoring and ranking."""
from __future__ import annotations

import pytest

from inference_x.benchmarks.advisor import ModelAdvisor
from inference_x.benchmarks.schemas import (
    AdvisorResult,
    BenchmarkResult,
    HardwareProfile,
    PromptResult,
)


def _rank(advisor: ModelAdvisor, hw: HardwareProfile, results: list[BenchmarkResult]) -> list[AdvisorResult]:
    return advisor.rank(hw, results).ranked


def _make_result(
    model_name: str,
    throughput: float,
    ttft_ms: float,
    vram_device_occupied_gib: float,
    *,
    hardware: HardwareProfile | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        suite_version="test",
        timestamp="2026-06-08T12:00:00+00:00",
        concurrency=1,
        prompt_results=[
            PromptResult(
                prompt_label="q1",
                tokens_generated=20,
                ttft_ms=ttft_ms,
                total_latency_ms=800.0,
                tokens_per_sec=throughput,
            )
        ],
        p50_latency_ms=800.0,
        p95_latency_ms=850.0,
        p99_latency_ms=900.0,
        mean_throughput_tps=throughput,
        vram_device_occupied_gib=vram_device_occupied_gib,
        hardware=hardware,
    )


def _hw(vram_free: float, vram_total: float, has_gpu: bool = True) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        cpu_cores=8,
        ram_total_gb=32.0,
        has_gpu=has_gpu,
    )


class TestModelAdvisorRanking:
    def test_higher_throughput_ranks_first(self):
        results = [
            _make_result("slow-model", throughput=10.0, ttft_ms=300.0, vram_device_occupied_gib=1.0),
            _make_result("fast-model", throughput=50.0, ttft_ms=100.0, vram_device_occupied_gib=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].model_name == "fast-model"
        assert ranked[1].model_name == "slow-model"

    def test_vram_exceeded_marks_not_viable(self):
        results = [
            _make_result("big-model", throughput=60.0, ttft_ms=50.0, vram_device_occupied_gib=8.0),
            _make_result("small-model", throughput=20.0, ttft_ms=200.0, vram_device_occupied_gib=1.5),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)

        big = next(r for r in ranked if r.model_name == "big-model")
        small = next(r for r in ranked if r.model_name == "small-model")

        assert big.viable is False
        assert big.score == 0.0
        assert small.viable is True
        assert small.score > 0.0

    def test_non_viable_model_ranks_last(self):
        results = [
            _make_result("too-big", throughput=100.0, ttft_ms=10.0, vram_device_occupied_gib=10.0),
            _make_result("fits", throughput=30.0, ttft_ms=200.0, vram_device_occupied_gib=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[-1].model_name == "too-big"

    def test_cpu_only_hardware_skips_vram_gate(self):
        results = [
            _make_result("model-a", throughput=10.0, ttft_ms=500.0, vram_device_occupied_gib=8.0),
        ]
        advisor = ModelAdvisor()
        hw = _hw(vram_free=0.0, vram_total=0.0, has_gpu=False)
        ranked = _rank(advisor, hw, results)
        assert ranked[0].viable is True
        assert ranked[0].score > 0.0

    def test_empty_results_returns_empty_list(self):
        advisor = ModelAdvisor()
        report = advisor.rank(_hw(vram_free=6.0, vram_total=8.0), [])
        assert report.ranked == []

    def test_recommendation_str_viable_format(self):
        results = [
            _make_result("qwen", throughput=42.0, ttft_ms=150.0, vram_device_occupied_gib=2.1),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        rec = ranked[0].recommendation_str
        assert "qwen" in rec
        assert "tok/s" in rec
        assert "GB VRAM" in rec

    def test_recommendation_str_not_viable_format(self):
        results = [
            _make_result("llama-big", throughput=80.0, ttft_ms=50.0, vram_device_occupied_gib=9.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        rec = ranked[0].recommendation_str
        assert "Skip" in rec or "skip" in rec or "requires" in rec

    def test_24gb_hardware_accepts_large_model(self):
        results = [
            _make_result("llama-8b", throughput=25.0, ttft_ms=200.0, vram_device_occupied_gib=7.5),
            _make_result("qwen-small", throughput=50.0, ttft_ms=80.0, vram_device_occupied_gib=1.2),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=20.0, vram_total=24.0), results)
        assert all(r.viable is True for r in ranked)

    def test_6gb_hardware_rejects_large_model(self):
        results = [
            _make_result("llama-8b", throughput=25.0, ttft_ms=200.0, vram_device_occupied_gib=7.5),
            _make_result("qwen-small", throughput=50.0, ttft_ms=80.0, vram_device_occupied_gib=1.2),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=4.0, vram_total=6.0), results)
        llama = next(r for r in ranked if r.model_name == "llama-8b")
        qwen = next(r for r in ranked if r.model_name == "qwen-small")
        assert llama.viable is False
        assert qwen.viable is True

    def test_score_is_a_within_report_ordinal(self):
        """Score ranks viable models within one report; it is not absolute quality.

        The bound 0–100 is a display artefact, not a portable quality scale — the
        same model can score differently in a different report (DEC-056). The
        assertions here only check the ordinal contract and the display bound.
        """
        results = [
            _make_result("model-a", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=1.0),
            _make_result("model-b", throughput=20.0, ttft_ms=300.0, vram_device_occupied_gib=2.0),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        for r in ranked:
            assert 0.0 <= r.score <= 100.0
        # Ordinal contract: better measured model ranks first.
        assert ranked[0].model_name == "model-a"

    def test_warm_ttft_skips_cold_start_prompt(self):
        cold = PromptResult(
            prompt_label="q1",
            tokens_generated=20,
            ttft_ms=1456.0,
            total_latency_ms=2000.0,
            tokens_per_sec=10.0,
        )
        warm = PromptResult(
            prompt_label="q2",
            tokens_generated=20,
            ttft_ms=13.0,
            total_latency_ms=800.0,
            tokens_per_sec=40.0,
        )
        results = [
            BenchmarkResult(
                model_name="qwen",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                vram_device_occupied_gib=1.0,
                prompt_results=[cold, warm],
            ),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].ttft_ms == 13.0


class TestVramSafetyBuffer:
    def test_marginal_footprint_not_viable_with_buffer(self):
        """5.6 GB footprint + 0.5 GB buffer = 6.1 GB > 6.0 GB total — must not pass gate."""
        results = [
            _make_result("marginal", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=5.6),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=3.3, vram_total=6.0), results)
        assert ranked[0].viable is False
        assert ranked[0].score == 0.0
        assert "5.60 GB + 0.5 GB buffer = 6.10 GB" in ranked[0].recommendation_str

    def test_footprint_within_total_is_viable(self):
        """5.38 GB footprint + 0.5 GB buffer = 5.88 GB <= 6.0 GB total — must pass gate."""
        results = [
            _make_result("qwen", throughput=40.0, ttft_ms=13.0, vram_device_occupied_gib=5.38),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=0.5, vram_total=6.0), results)
        assert ranked[0].viable is True
        assert ranked[0].score > 0.0


class TestModelAdvisorHardware:
    def test_skips_mismatched_hardware(self, make_hardware):
        desktop = make_hardware(gpu_name="RTX 4090", vram_total_gb=24.0, vram_free_gb=20.0)
        laptop = make_hardware(gpu_name="RTX 3060 Laptop GPU", vram_total_gb=6.0, vram_free_gb=4.9)
        results = [
            _make_result(
                "big-model",
                throughput=50.0,
                ttft_ms=100.0,
                vram_device_occupied_gib=2.0,
                hardware=desktop,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(laptop, results)
        assert report.ranked == []
        assert any("Skipped big-model" in w for w in report.warnings)

    def test_accepts_matching_hardware(self, make_hardware):
        hw = make_hardware()
        results = [
            _make_result(
                "small-model",
                throughput=30.0,
                ttft_ms=150.0,
                vram_device_occupied_gib=1.5,
                hardware=hw,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(hw, results)
        assert len(report.ranked) == 1
        assert report.ranked[0].model_name == "small-model"
        assert not any("Skipped" in w for w in report.warnings)

    def test_rejects_substring_gpu_name_match(self, make_hardware):
        """RTX 3060 benchmark must not match RTX 3060 Ti (different GPU)."""
        rtx3060 = make_hardware(gpu_name="RTX 3060", vram_total_gb=6.0, vram_free_gb=4.9)
        rtx3060ti = make_hardware(gpu_name="RTX 3060 Ti", vram_total_gb=8.0, vram_free_gb=6.0)
        results = [
            _make_result(
                "cached-model",
                throughput=40.0,
                ttft_ms=100.0,
                vram_device_occupied_gib=2.0,
                hardware=rtx3060,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(rtx3060ti, results)
        assert report.ranked == []
        assert any("Skipped cached-model" in w for w in report.warnings)

    def test_legacy_result_without_hardware_warns_but_ranks(self):
        results = [
            _make_result("legacy-model", throughput=25.0, ttft_ms=120.0, vram_device_occupied_gib=1.0),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(_hw(vram_free=6.0, vram_total=8.0), results)
        assert len(report.ranked) == 1
        assert any("No hardware recorded" in w for w in report.warnings)


class TestAdvisorMaxModelLenWarning:
    def test_warns_on_mismatch(self, make_hardware):
        hw = make_hardware()
        results = [
            BenchmarkResult(
                model_name="qwen2.5-0.5b",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                vram_device_occupied_gib=1.5,
                hardware=hw,
                max_model_len=4096,
                prompt_results=[
                    PromptResult(
                        prompt_label="q1",
                        tokens_generated=20,
                        ttft_ms=100.0,
                        total_latency_ms=800.0,
                        tokens_per_sec=40.0,
                    )
                ],
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(
            hw,
            results,
            model_max_lens={"qwen2.5-0.5b": 8192},
        )
        assert any("max_model_len=4096" in w and "8192" in w for w in report.warnings)
        assert len(report.ranked) == 1

    def test_no_warning_when_matching(self, make_hardware):
        hw = make_hardware()
        results = [
            BenchmarkResult(
                model_name="qwen2.5-0.5b",
                suite_version="test",
                timestamp="2026-06-08T12:00:00+00:00",
                mean_throughput_tps=40.0,
                vram_device_occupied_gib=1.5,
                hardware=hw,
                max_model_len=8192,
            ),
        ]
        advisor = ModelAdvisor()
        report = advisor.rank(
            hw,
            results,
            model_max_lens={"qwen2.5-0.5b": 8192},
        )
        assert not any("max_model_len" in w for w in report.warnings)


class TestHonestScoring:
    """DEC-056: measured-only score, exact weights, `viable` as sole signal."""

    def test_viable_worst_and_gated_both_zero_distinct_by_viable(self):
        """A viable-worst model and a VRAM-gated model can both score 0.0.

        They are distinguished only by `viable`, never by the score value (V7).
        """
        results = [
            _make_result("good", throughput=100.0, ttft_ms=10.0, vram_device_occupied_gib=1.0),
            # Worst on every measured axis but still fits: throughput 0 → 0,
            # ttft == max → 0, headroom == 0 at the viability boundary → 0.
            _make_result("worst-viable", throughput=0.0, ttft_ms=1000.0, vram_device_occupied_gib=9.5),
            # Just over the buffer boundary → hard-gated.
            _make_result("gated", throughput=80.0, ttft_ms=20.0, vram_device_occupied_gib=9.6),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=5.0, vram_total=10.0), results)
        by_name = {r.model_name: r for r in ranked}

        assert by_name["worst-viable"].score == 0.0
        assert by_name["worst-viable"].viable is True
        assert by_name["gated"].score == 0.0
        assert by_name["gated"].viable is False

    def test_no_constant_floor(self):
        """With `quant_score` gone, a single viable model has no +10 floor.

        Single result → TTFT component is 0 (it is its own max), throughput
        component is 1; the score is exactly the measured composition.
        """
        results = [
            _make_result("solo", throughput=42.0, ttft_ms=100.0, vram_device_occupied_gib=0.5),
        ]
        advisor = ModelAdvisor()
        ranked = _rank(advisor, _hw(vram_free=50.0, vram_total=100.0), results)
        vram_required = 0.5 + 0.5  # buffer
        vram_score = (100.0 - vram_required) / 100.0
        expected = ((4 / 9) * 1.0 + (1 / 3) * 0.0 + (2 / 9) * vram_score) * 100.0
        assert ranked[0].score == pytest.approx(round(expected, 2), abs=0.01)

    def test_adding_a_model_changes_other_scores(self):
        """Relative normalisation: a model's score is not portable across reports."""
        advisor = ModelAdvisor()
        hw = _hw(vram_free=6.0, vram_total=8.0)
        solo = _make_result("a", throughput=30.0, ttft_ms=100.0, vram_device_occupied_gib=1.0)

        score_alone = _rank(advisor, hw, [solo])[0].score
        faster = _make_result("b", throughput=90.0, ttft_ms=300.0, vram_device_occupied_gib=1.0)
        with_b = {r.model_name: r for r in _rank(advisor, hw, [solo, faster])}
        assert with_b["a"].score != score_alone

    def test_viable_ranking_invariant_under_reweighting(self):
        """Re-normalising the three weights preserves the viable ordering (V3).

        The measured-only ordering (prior 0.40 : 0.30 : 0.20 ratio) equals the
        ordering under the exact fractions, because the new weights are an affine
        scaling of the old measured ones. Verified against an independent
        computation of the component scores.
        """
        results = [
            _make_result("m1", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=1.0),
            _make_result("m2", throughput=80.0, ttft_ms=50.0, vram_device_occupied_gib=2.0),
            _make_result("m3", throughput=20.0, ttft_ms=200.0, vram_device_occupied_gib=1.5),
        ]
        hw = _hw(vram_free=20.0, vram_total=24.0)
        advisor = ModelAdvisor()
        advisor_order = [r.model_name for r in _rank(advisor, hw, results)]

        max_t = max(r.mean_throughput_tps for r in results)
        max_tt = max(r.prompt_results[0].ttft_ms for r in results)

        def legacy_measured_score(r: BenchmarkResult) -> float:
            ts = r.mean_throughput_tps / max_t
            tts = 1.0 - r.prompt_results[0].ttft_ms / max_tt
            required = r.vram_device_occupied_gib + 0.5
            vs = max(0.0, min(1.0, (hw.vram_total_gb - required) / hw.vram_total_gb))
            return 0.40 * ts + 0.30 * tts + 0.20 * vs

        legacy_order = [
            r.model_name
            for r in sorted(results, key=legacy_measured_score, reverse=True)
        ]
        assert advisor_order == legacy_order


class TestScoreMonotonicity:
    """V4a: varying one measured component alone moves ranking that way."""

    def test_throughput_monotonic(self):
        results = [
            _make_result("hi", throughput=80.0, ttft_ms=100.0, vram_device_occupied_gib=1.0),
            _make_result("lo", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=1.0),
        ]
        ranked = _rank(ModelAdvisor(), _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].model_name == "hi"

    def test_ttft_monotonic(self):
        results = [
            _make_result("fast-ttft", throughput=40.0, ttft_ms=50.0, vram_device_occupied_gib=1.0),
            _make_result("slow-ttft", throughput=40.0, ttft_ms=250.0, vram_device_occupied_gib=1.0),
        ]
        ranked = _rank(ModelAdvisor(), _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].model_name == "fast-ttft"

    def test_vram_monotonic(self):
        results = [
            _make_result("lean", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=1.0),
            _make_result("heavy", throughput=40.0, ttft_ms=100.0, vram_device_occupied_gib=3.0),
        ]
        ranked = _rank(ModelAdvisor(), _hw(vram_free=6.0, vram_total=8.0), results)
        assert ranked[0].model_name == "lean"


class TestVramAliasAndSerialization:
    """DEC-057: canonical VRAM field with deprecated aliases through Phase A."""

    def test_advisor_reads_peak_vram_delta_alias(self):
        """A historical result carrying `peak_vram_delta_gb` is scored via alias."""
        legacy = BenchmarkResult.model_validate(
            {
                "model_name": "legacy",
                "suite_version": "test",
                "timestamp": "2026-06-08T12:00:00+00:00",
                "mean_throughput_tps": 30.0,
                "peak_vram_delta_gb": 2.0,
                "prompt_results": [
                    {
                        "prompt_label": "q1",
                        "tokens_generated": 20,
                        "ttft_ms": 100.0,
                        "total_latency_ms": 800.0,
                        "tokens_per_sec": 30.0,
                    }
                ],
            }
        )
        assert legacy.vram_device_occupied_gib == 2.0
        ranked = _rank(ModelAdvisor(), _hw(vram_free=6.0, vram_total=8.0), [legacy])
        assert ranked[0].vram_device_occupied_gib == 2.0

    def test_advisor_result_serialises_canonical_and_alias(self):
        """AdvisorResult exposes the canonical field and still serialises `vram_gb`."""
        results = [
            _make_result("m", throughput=30.0, ttft_ms=100.0, vram_device_occupied_gib=1.5),
        ]
        ranked = _rank(ModelAdvisor(), _hw(vram_free=6.0, vram_total=8.0), results)
        dumped = ranked[0].model_dump()
        assert dumped["vram_device_occupied_gib"] == 1.5
        assert dumped["vram_gb"] == 1.5

    def test_advisor_result_canonical_wins_on_conflict(self):
        """Conflicting canonical + alias input resolves to the canonical value."""
        r = AdvisorResult.model_validate(
            {
                "model_name": "m",
                "score": 10.0,
                "viable": True,
                "throughput_tps": 30.0,
                "ttft_ms": 100.0,
                "vram_device_occupied_gib": 1.5,
                "vram_gb": 9.9,
                "recommendation_str": "m",
            }
        )
        assert r.vram_device_occupied_gib == 1.5
        assert r.vram_gb == 1.5
