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

        return MetricsSummary(
            total_requests=len(records),
            error_count=errors,
            avg_latency_ms=round(avg, 3),
            p95_latency_ms=round(p95, 3),
        )
