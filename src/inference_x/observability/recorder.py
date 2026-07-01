"""MetricsRecorder: the single write path for all observability data.

Middleware calls recorder.record(...) after every request.  The recorder
writes to InMemoryStorage and optionally exports via a configured exporter.
Neither call should raise; errors are logged at WARNING and swallowed so they
never propagate into the request path.
"""
from __future__ import annotations

import logging
import uuid

from inference_x.observability.exporters import NullExporter
from inference_x.observability.storage import InMemoryStorage, RequestRecord

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """Records one RequestRecord per request into storage and an optional exporter.

    This is the only class that constructs RequestRecord objects.  Callers
    provide raw values; the recorder handles ID generation and error safety.
    """

    def __init__(
        self,
        storage: InMemoryStorage,
        exporter: NullExporter | None = None,
    ) -> None:
        self._storage = storage
        self._exporter = exporter or NullExporter()

    def record(
        self,
        *,
        path: str,
        method: str,
        status_code: int,
        latency_ms: float,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        error: bool = False,
        ttft_ms: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> RequestRecord:
        """Build, store, and export a RequestRecord.

        *ttft_ms* and *tokens_per_sec* are only populated for streaming chat
        completions (see ObservabilityMiddleware); every other request leaves
        them None.

        Always returns the record even if storage or export fails.
        """
        record = RequestRecord(
            request_id=uuid.uuid4().hex,
            path=path,
            method=method,
            status_code=status_code,
            latency_ms=round(latency_ms, 3),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=error,
            ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
            tokens_per_sec=round(tokens_per_sec, 3) if tokens_per_sec is not None else None,
        )
        try:
            self._storage.append(record)
        except Exception as exc:
            logger.warning("MetricsRecorder: storage.append failed: %s", exc)

        try:
            self._exporter.export(record)
        except Exception as exc:
            logger.warning("MetricsRecorder: exporter.export failed: %s", exc)

        return record

    @property
    def storage(self) -> InMemoryStorage:
        return self._storage
