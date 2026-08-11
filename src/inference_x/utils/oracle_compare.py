"""Teacher-forced oracle comparison helpers (Phase C — C4a).

Pure classification of per-position top-1 agreement vs a transformers oracle.
Near-ties are reported, never silently counted as exact agreement
(REVIEW-2026-08-03 §6.2b; add-oracle-conformance design.md D3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


DEFAULT_NEAR_TIE_EPS = 1e-3


class PositionClass(str, Enum):
    EXACT = "exact"
    NEAR_TIE = "near_tie"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class PositionScore:
    """Top-1 / top-2 ids and their logit (or logprob) values at one position."""

    top1_id: int
    top1_logit: float
    top2_id: int | None = None
    top2_logit: float | None = None

    @property
    def margin(self) -> float | None:
        if self.top2_logit is None:
            return None
        return abs(self.top1_logit - self.top2_logit)


@dataclass(frozen=True, slots=True)
class PositionVerdict:
    index: int
    classification: PositionClass
    engine: PositionScore
    reference: PositionScore


@dataclass(frozen=True, slots=True)
class OracleReport:
    verdicts: tuple[PositionVerdict, ...]

    @property
    def exact_count(self) -> int:
        return sum(1 for v in self.verdicts if v.classification is PositionClass.EXACT)

    @property
    def near_ties(self) -> tuple[PositionVerdict, ...]:
        return tuple(v for v in self.verdicts if v.classification is PositionClass.NEAR_TIE)

    @property
    def mismatches(self) -> tuple[PositionVerdict, ...]:
        return tuple(v for v in self.verdicts if v.classification is PositionClass.MISMATCH)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _is_near_tie(score: PositionScore, eps: float) -> bool:
    margin = score.margin
    return margin is not None and margin < eps


def classify_position(
    index: int,
    engine: PositionScore,
    reference: PositionScore,
    *,
    eps: float = DEFAULT_NEAR_TIE_EPS,
) -> PositionVerdict:
    """Classify one teacher-forced position.

    - exact: both sides share the same top-1 id and neither side is a near-tie
      (a near-tie agreement is still reported as near_tie — never a silent pass).
    - near_tie: top-1 agrees but either side has a sub-eps margin, OR top-1
      disagrees while at least one side has a sub-eps margin.
    - mismatch: top-1 disagrees and neither side is a near-tie.
    """
    agree = engine.top1_id == reference.top1_id
    near = _is_near_tie(engine, eps) or _is_near_tie(reference, eps)
    if near:
        # Near-ties are always reported — never counted as silent exact agreement.
        cls = PositionClass.NEAR_TIE
    elif agree:
        cls = PositionClass.EXACT
    else:
        cls = PositionClass.MISMATCH
    return PositionVerdict(
        index=index, classification=cls, engine=engine, reference=reference
    )


def classify_positions(
    engine_scores: list[PositionScore],
    reference_scores: list[PositionScore],
    *,
    eps: float = DEFAULT_NEAR_TIE_EPS,
) -> OracleReport:
    if len(engine_scores) != len(reference_scores):
        raise ValueError(
            f"score length mismatch: engine={len(engine_scores)} "
            f"reference={len(reference_scores)}"
        )
    verdicts = tuple(
        classify_position(i, eng, ref, eps=eps)
        for i, (eng, ref) in enumerate(zip(engine_scores, reference_scores, strict=True))
    )
    return OracleReport(verdicts=verdicts)


def format_report(report: OracleReport) -> str:
    lines = [
        f"oracle: exact={report.exact_count}/{len(report.verdicts)} "
        f"near_ties={len(report.near_ties)} mismatches={len(report.mismatches)}",
    ]
    for v in report.near_ties:
        lines.append(
            f"  near_tie@{v.index}: engine={v.engine.top1_id} "
            f"(margin={v.engine.margin}) ref={v.reference.top1_id} "
            f"(margin={v.reference.margin})"
        )
    for v in report.mismatches:
        lines.append(
            f"  mismatch@{v.index}: engine={v.engine.top1_id} ref={v.reference.top1_id}"
        )
    return "\n".join(lines)
