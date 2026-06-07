from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestRecord:
    """One captured inference request.

    Fields are deliberately flat so they can be serialised to JSON or
    inspected in tests without traversing nested structures.
    """

    request_id: str
    path: str
    method: str
    status_code: int
    latency_ms: float
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemoryStorage:
    """Thread-safe, capped ring-buffer of RequestRecord objects.

    Once *max_records* is reached the oldest entry is dropped to make room.
    All reads and writes hold a short lock; no I/O touches the request path.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._buf: deque[RequestRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def append(self, record: RequestRecord) -> None:
        with self._lock:
            self._buf.append(record)

    def all(self) -> list[RequestRecord]:
        with self._lock:
            return list(self._buf)

    def recent(self, n: int) -> list[RequestRecord]:
        with self._lock:
            items = list(self._buf)
        return items[-n:] if n <= len(items) else items

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
