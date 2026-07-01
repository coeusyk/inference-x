from __future__ import annotations

from dataclasses import dataclass

from inference_x.observability.recorder import MetricsRecorder
from inference_x.observability.storage import RequestRecord


@dataclass
class MetricsSummary:
    total_requests: int
    error_count: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    avg_ttft_ms: float | None = None
    avg_tokens_per_sec: float | None = None


class MetricsService:
    """Read-only inspection of recorded request metrics.

    Used for testing and future inspection endpoints.
    Does not own storage — reads through the recorder's storage reference.
    """

    def __init__(self, recorder: MetricsRecorder) -> None:
        self._recorder = recorder

    def recent(self, n: int = 100) -> list[RequestRecord]:
        return self._recorder.storage.recent(n)

    def all(self) -> list[RequestRecord]:
        return self._recorder.storage.all()

    def summary(self) -> MetricsSummary:
        records = self._recorder.storage.all()
        if not records:
            return MetricsSummary(
                total_requests=0,
                error_count=0,
                avg_latency_ms=None,
                p95_latency_ms=None,
            )

        latencies = sorted(r.latency_ms for r in records)
        avg = sum(latencies) / len(latencies)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = latencies[p95_idx]
        errors = sum(1 for r in records if r.error)

        # TTFT/tokens-per-sec are only populated on streaming chat completions
        # (see ObservabilityMiddleware); average over just those records so a
        # mix of streaming and non-streaming traffic doesn't skew the mean
        # toward None-as-zero.
        ttfts = [r.ttft_ms for r in records if r.ttft_ms is not None]
        tps = [r.tokens_per_sec for r in records if r.tokens_per_sec is not None]

        return MetricsSummary(
            total_requests=len(records),
            error_count=errors,
            avg_latency_ms=round(avg, 3),
            p95_latency_ms=round(p95, 3),
            avg_ttft_ms=round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            avg_tokens_per_sec=round(sum(tps) / len(tps), 3) if tps else None,
        )
