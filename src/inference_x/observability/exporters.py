"""Exporters write completed RequestRecords out of process.

The default exporter is NullExporter (no-op).  JsonLineExporter appends one
NDJSON line per record to a file — enabled by setting the env var
INFERENCE_X_METRICS_FILE to a writable path.

Exporters are called from the recorder after storage is updated, still within
the middleware coroutine but after the response has been sent.  They should
not raise; errors are swallowed and logged at WARNING level.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from inference_x.observability.storage import RequestRecord

logger = logging.getLogger(__name__)


class NullExporter:
    """No-op exporter. Default when INFERENCE_X_METRICS_FILE is not set."""

    def export(self, record: RequestRecord) -> None:  # noqa: ARG002
        pass


class JsonLineExporter:
    """Appends one NDJSON line per record to *file_path*.

    Opens in append mode on each write to avoid holding a file handle
    across idle periods.  File is created if it does not exist.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)

    def export(self, record: RequestRecord) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("JsonLineExporter: failed to write record: %s", exc)


def build_exporter() -> NullExporter | JsonLineExporter:
    """Return the configured exporter based on environment variables.

    INFERENCE_X_METRICS_FILE — if set and non-empty, use JsonLineExporter
                               with that path.  Otherwise use NullExporter.
    """
    metrics_file = os.environ.get("INFERENCE_X_METRICS_FILE", "").strip()
    if metrics_file:
        logger.info("Metrics exporter: JsonLineExporter → %s", metrics_file)
        return JsonLineExporter(metrics_file)
    return NullExporter()
