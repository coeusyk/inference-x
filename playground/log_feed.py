"""Format and tail playground server logs for the loading screen."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = _REPO_ROOT / "logs" / "playground-server.log"

_SESSION_MARKER = "--- playground session ---"
_LEVEL_MARKERS = {
    "CRITICAL": "✗",
    "ERROR": "✗",
    "WARNING": "!",
    "INFO": "·",
    "DEBUG": "·",
}

_SKIP_PREFIXES = (
    "Traceback (most recent call last)",
    "  File ",
    'File "',
    "    ",
    "^",
    "The above exception",
    "During handling",
)

_STRUCTURED_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(CRITICAL|INFO|WARNING|ERROR|DEBUG)\] (.+)$"
)
_VLLM_RE = re.compile(
    r"^(?:\(EngineCore pid=\d+\) )?(INFO|WARNING|ERROR) \d{2}-\d{2} \d{2}:\d{2}:\d{2} \[[^\]]+\] (.+)$"
)


def _shorten_logger_message(message: str, *, max_len: int = 66) -> str | None:
    """Map known inference_x / uvicorn messages to short feed lines."""
    if message.startswith("Started server process"):
        return "Server process started"
    if message.startswith("Waiting for application startup"):
        return "Application startup…"
    if "Initializing InferenceX" in message:
        return "Starting InferenceX"
    if message.startswith("Model registry loaded"):
        return "Model registry loaded"
    if message.startswith("Task router ready"):
        return "Task router ready"
    if m := re.search(r"Loading engine for model=([^\s(]+)", message):
        return f"Loading engine: {m.group(1)}"
    if m := re.search(
        r"GPU memory for model=([^:]+): utilization=([\d.]+).*free=([\d.]+) GiB.*total=([\d.]+) GiB",
        message,
    ):
        util_pct = int(float(m.group(2)) * 100)
        return f"GPU {m.group(3)}/{m.group(4)} GiB free (util {util_pct}%)"
    if "Initializing vLLM engine for model=" in message:
        model = message.split("model=", 1)[-1].split()[0]
        return f"Starting vLLM: {model}"
    if message.startswith("Loading weights"):
        return "Downloading/loading weights…"
    if message.startswith("CUDA_HOME set"):
        return "CUDA toolkit ready"
    if "Startup initialization failed:" in message:
        return message.split("Startup initialization failed:", 1)[-1].strip()[:max_len]
    if message.startswith("vLLM engine ready:"):
        return f"Engine ready: {message.split('model=', 1)[-1].split()[0]}"
    if message.startswith("Engine pool ready:"):
        return "Engine pool ready"
    if message == "Application startup complete.":
        return "Application startup complete"
    if message.startswith("Uvicorn running on"):
        host = message.split("on ", 1)[-1].split(" (", 1)[0]
        return f"Listening on {host}"
    if message.startswith("Application startup failed"):
        return "Application startup failed"
    if message.startswith("Traceback"):
        return None
    return message[:max_len]


def _shorten_vllm_message(message: str, *, max_len: int = 66) -> str | None:
    if "Starting to load model" in message:
        return "Loading model weights…"
    if "Loading weights took" in message:
        took = message.split("took", 1)[-1].strip()
        return f"Weights loaded ({took[:28]})"
    if "Model loading took" in message:
        took = message.split("took", 1)[-1].strip()
        return f"Model loaded ({took[:28]})"
    if "Graph capturing finished" in message:
        return "CUDA graphs ready"
    if "init engine" in message and "took" in message:
        took = message.split("took", 1)[-1].strip()
        return f"vLLM warmup done ({took[:22]})"
    if message.startswith("Available KV cache memory:"):
        val = message.split(":", 1)[-1].strip()
        if val.startswith("-"):
            return f"No KV cache memory ({val})"
        return f"KV cache {val[:30]}"
    if "Using FLASH_ATTN" in message:
        return "Using FlashAttention"
    if "Resolved architecture:" in message:
        arch = message.split(":", 1)[-1].strip()
        return f"Architecture: {arch}"
    if "WSL is detected" in message:
        return "WSL detected (pin_memory off)"
    if "No available memory for the cache blocks" in message:
        return "No KV cache memory — increase GPU budget"
    return None


def format_log_line(line: str, *, max_len: int = 66) -> str | None:
    """Return a short display line for the loading feed, or None to skip."""
    raw = line.strip()
    if not raw or raw == _SESSION_MARKER:
        return None
    if raw.startswith(_SKIP_PREFIXES):
        return None
    if raw.startswith("RuntimeError:") or raw.startswith("ValueError:"):
        err = raw.split(":", 1)[-1].strip()
        return f"✗ {err[:max_len]}"

    if "Loading safetensors checkpoint shards:" in raw and "100%" in raw:
        return "· Model weights loaded"
    if raw.startswith("Capturing CUDA graphs") and "100%" in raw:
        return "· CUDA graphs captured"
    if "Capturing CUDA graphs" in raw and "%|" in raw:
        return None

    structured = _STRUCTURED_RE.match(raw)
    if structured:
        level, rest = structured.group(1), structured.group(2)
        _, _, message = rest.partition(": ")
        short = _shorten_logger_message(message or rest, max_len=max_len)
        if short is None:
            return None
        prefix = _LEVEL_MARKERS.get(level, "·")
        return f"{prefix} {short}"[: max_len + 2]

    vllm = _VLLM_RE.match(raw)
    if vllm:
        level, msg = vllm.group(1), vllm.group(2)
        if level == "ERROR" or "ValueError:" in msg:
            short = _shorten_vllm_message(msg, max_len=max_len) or msg.split("ValueError:", 1)[-1].strip()
            return f"✗ {short[:max_len]}"
        short = _shorten_vllm_message(msg, max_len=max_len)
        if short:
            return f"· {short}"[: max_len + 2]
        return None

    if raw.upper().startswith("WARNING"):
        msg = raw.split(":", 1)[-1].strip() if ":" in raw else raw[7:].strip()
        return f"! {msg[:max_len]}"
    return None


def extract_error_summary(log_path: Path = DEFAULT_LOG_PATH, *, tail_lines: int = 80) -> str:
    """Return a one-line error summary from the end of the server log."""
    if not log_path.is_file():
        return "Server log not found — check logs/playground-server.log"

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "Could not read server log"

    for line in reversed(lines[-tail_lines:]):
        structured = _STRUCTURED_RE.match(line.strip())
        if structured and structured.group(1) in ("CRITICAL", "ERROR"):
            rest = structured.group(2)
            _, _, message = rest.partition(": ")
            if "Startup initialization failed:" in (message or rest):
                return (message or rest).split("Startup initialization failed:", 1)[-1].strip()[:200]
            return (message or rest)[:200]

    for line in reversed(lines[-tail_lines:]):
        stripped = line.strip()
        if "RuntimeError:" in stripped:
            return stripped.split("RuntimeError:", 1)[-1].strip()[:200]
        if "ValueError:" in stripped:
            return stripped.split("ValueError:", 1)[-1].strip()[:200]

    return "Model failed to load — see log lines above"


def startup_failed_in_log(log_path: Path = DEFAULT_LOG_PATH) -> bool:
    """Return True when the server log shows a fatal startup failure."""
    if not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        "Application startup failed" in text
        or "[CRITICAL]" in text
        or "Startup initialization failed" in text
    )


def prepare_log_session(log_path: Path = DEFAULT_LOG_PATH) -> int:
    """Overwrite the server log for a new playground session; return tail offset."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"{_SESSION_MARKER}\n", encoding="utf-8")
    return log_path.stat().st_size


# Backwards-compatible alias for tests
mark_log_session = prepare_log_session


class LogTailer:
    """Poll a log file and emit formatted lines via callback."""

    def __init__(
        self,
        on_line: Callable[[str], None],
        *,
        log_path: Path = DEFAULT_LOG_PATH,
        poll_interval_s: float = 0.35,
    ) -> None:
        self._on_line = on_line
        self._log_path = log_path
        self._poll_interval_s = poll_interval_s
        self._offset = 0
        self._task: asyncio.Task[None] | None = None
        self._buffer = ""

    def start(self, *, from_offset: int | None = None) -> None:
        if from_offset is not None:
            self._offset = from_offset
        elif self._log_path.is_file():
            self._offset = self._log_path.stat().st_size
        else:
            self._offset = 0
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await self._poll_once()
                await asyncio.sleep(self._poll_interval_s)
        except asyncio.CancelledError:
            await self._poll_once()
            raise

    async def _poll_once(self) -> None:
        if not self._log_path.is_file():
            return
        try:
            with self._log_path.open(encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return

        if not chunk:
            return

        self._buffer += chunk
        lines = self._buffer.splitlines()
        self._buffer = lines.pop() if lines and not chunk.endswith("\n") else ""

        for line in lines:
            formatted = format_log_line(line)
            if formatted:
                self._on_line(formatted)
