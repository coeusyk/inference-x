"""Unit tests for C4a oracle near-tie / agreement classification (GPU-free)."""

from __future__ import annotations

import pytest

from inference_x.utils.oracle_compare import (
    PositionClass,
    PositionScore,
    classify_position,
    classify_positions,
    format_report,
)


def _score(top1: int, top1_logit: float, top2: int | None = None, top2_logit: float | None = None) -> PositionScore:
    return PositionScore(
        top1_id=top1,
        top1_logit=top1_logit,
        top2_id=top2,
        top2_logit=top2_logit,
    )


def test_exact_agreement_when_top1_matches_with_clear_margin():
    verdict = classify_position(
        0,
        _score(10, 5.0, 11, 1.0),
        _score(10, 4.5, 12, 0.5),
        eps=1e-3,
    )
    assert verdict.classification is PositionClass.EXACT


def test_near_tie_when_agreement_but_margin_below_eps():
    """Near-ties must be reported, not silently counted as exact (§6.2b / D3)."""
    verdict = classify_position(
        3,
        _score(7, 1.0005, 8, 1.0000),
        _score(7, 2.0, 9, 0.1),
        eps=1e-3,
    )
    assert verdict.classification is PositionClass.NEAR_TIE


def test_near_tie_on_disagreement_with_small_margin():
    verdict = classify_position(
        1,
        _score(1, 0.5, 2, 0.4995),
        _score(2, 0.5, 1, 0.4995),
        eps=1e-3,
    )
    assert verdict.classification is PositionClass.NEAR_TIE


def test_hard_mismatch_when_disagree_without_near_tie():
    verdict = classify_position(
        2,
        _score(1, 3.0, 2, 0.1),
        _score(9, 3.0, 8, 0.1),
        eps=1e-3,
    )
    assert verdict.classification is PositionClass.MISMATCH


def test_report_aggregates_and_pass_fail():
    report = classify_positions(
        [
            _score(1, 5.0, 2, 0.1),
            _score(3, 1.0004, 4, 1.0000),
            _score(5, 2.0, 6, 0.1),
        ],
        [
            _score(1, 4.0, 2, 0.2),
            _score(3, 2.0, 9, 0.1),
            _score(7, 2.0, 8, 0.1),
        ],
        eps=1e-3,
    )
    assert report.exact_count == 1
    assert len(report.near_ties) == 1
    assert report.near_ties[0].index == 1
    assert len(report.mismatches) == 1
    assert report.mismatches[0].index == 2
    assert report.passed is False

    text = format_report(report)
    assert "exact=1/3" in text
    assert "near_tie@1" in text
    assert "mismatch@2" in text


def test_report_passes_with_only_exact_and_near_ties():
    report = classify_positions(
        [_score(1, 5.0, 2, 0.1), _score(3, 1.0004, 4, 1.0000)],
        [_score(1, 4.0, 2, 0.2), _score(3, 2.0, 9, 0.1)],
        eps=1e-3,
    )
    assert report.passed is True
    assert report.exact_count == 1
    assert len(report.near_ties) == 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="score length mismatch"):
        classify_positions([_score(1, 1.0)], [_score(1, 1.0), _score(2, 1.0)])
