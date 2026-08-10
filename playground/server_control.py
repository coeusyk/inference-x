"""Start/stop the playground inference server(s) with selected models loaded.

B6: each server process serves exactly one model (Option A, one model per OS
process). Multi-model compare mode is achieved by launching one process per
model on consecutive ports, starting from the port in the caller's base_url.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs"
_LOG_PATH = _LOG_DIR / "playground-server.log"

_server_started_by_playground: bool = False


def playground_started_server() -> bool:
    """True when this process started the background uvicorn server(s)."""
    return _server_started_by_playground


def _reset_playground_server_state() -> None:
    """Clear the started-server flag (for tests)."""
    global _server_started_by_playground
    _server_started_by_playground = False


def _log_path_for(index: int) -> Path:
    """Log file for the Nth server process; index 0 keeps the original path."""
    if index == 0:
        return _LOG_PATH
    return _LOG_DIR / f"playground-server-{index}.log"


def _target_url(base_url: str, index: int) -> str:
    """Base URL for the Nth server process; index 0 is *base_url* unchanged."""
    if index == 0:
        return base_url
    parsed = httpx.URL(base_url)
    return str(parsed.copy_with(port=(parsed.port or 8000) + index))


async def fetch_loaded_models(base_url: str) -> list[str]:
    """Return ``loaded_models`` from ``GET /health``, or ``[]`` if unreachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/health")
            if resp.status_code == 200:
                loaded = resp.json().get("loaded_models")
                if isinstance(loaded, list):
                    return [str(m) for m in loaded]
    except httpx.HTTPError:
        pass
    return []


async def stop_playground_server() -> None:
    """Stop uvicorn and orphaned vLLM worker processes from prior playground runs."""
    for pattern in (
        "[u]vicorn inference_x.api.main:app",
        "VLLM::Engine[C]ore",
    ):
        proc = await asyncio.create_subprocess_exec(
            "pkill",
            "-f",
            pattern,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    await asyncio.sleep(1.5)


async def cleanup_playground_server_if_started() -> None:
    """Stop the server when the playground started it (no-op if user attached externally)."""
    global _server_started_by_playground
    if not _server_started_by_playground:
        return
    await stop_playground_server()
    _server_started_by_playground = False


def cleanup_playground_server_if_started_sync() -> None:
    """Synchronous wrapper for CLI exit handlers."""
    asyncio.run(cleanup_playground_server_if_started())


async def start_playground_server(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_path: Path = _LOG_PATH,
) -> None:
    """Launch uvicorn with ``INFERENCE_X_DEFAULT_MODEL`` set to *model*."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["INFERENCE_X_DEFAULT_MODEL"] = model
    env.pop("INFERENCE_X_LOADED_MODELS", None)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    log_f = log_path.open("a", encoding="utf-8")
    try:
        await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "uvicorn",
            "inference_x.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            cwd=_REPO_ROOT,
            env=env,
            stdout=log_f,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_f.close()


async def wait_for_health(
    base_url: str,
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 2.0,
    log_path: Path | None = None,
) -> bool:
    """Poll ``GET /health`` until HTTP 200, startup failure in log, or timeout."""
    try:
        from log_feed import startup_failed_in_log
    except ImportError:
        from playground.log_feed import startup_failed_in_log

    deadline = time.monotonic() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
        except httpx.HTTPError:
            pass
        if log_path is not None and startup_failed_in_log(log_path):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval_s, remaining))
    return False


async def ensure_models_loaded(
    base_url: str,
    models: list[str],
    *,
    on_status: Callable[[str], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    load_timeout_s: int = 600,
) -> dict[str, str] | None:
    """Start or restart one server process per model; each on its own port.

    Returns a ``{model: base_url}`` mapping on success, ``None`` on failure.
    ``models[0]`` is always served at *base_url* itself; each subsequent
    model gets its own process on ``base_url``'s port + its index.
    """
    global _server_started_by_playground
    try:
        from log_feed import LogTailer, extract_error_summary, prepare_log_session
    except ImportError:
        from playground.log_feed import LogTailer, extract_error_summary, prepare_log_session
    try:
        from streaming import wait_for_model
    except ImportError:
        from playground.streaming import wait_for_model

    wanted = list(dict.fromkeys(models))
    if not wanted:
        return None

    def status(message: str) -> None:
        if on_status is not None:
            on_status(message)

    def log_line(message: str) -> None:
        if on_log is not None:
            on_log(message)

    targets = [
        (model, _target_url(base_url, i), _log_path_for(i)) for i, model in enumerate(wanted)
    ]
    multi = len(targets) > 1

    for _, _, log_path in targets:
        prepare_log_session(log_path)

    def _on_line(model: str) -> Callable[[str], None]:
        return (lambda line: log_line(f"[{model}] {line}")) if multi else log_line

    tailers = [LogTailer(_on_line(model), log_path=log_path) for model, _, log_path in targets]
    for tailer in tailers:
        tailer.start(from_offset=0)

    try:
        already_loaded = True
        for model, url, _ in targets:
            if model not in await fetch_loaded_models(url):
                already_loaded = False
                break

        if not already_loaded:
            status(f"Starting server(s) for {', '.join(wanted)}…")
            await stop_playground_server()
            for model, url, log_path in targets:
                port = httpx.URL(url).port or 8000
                await start_playground_server(model, port=port, log_path=log_path)
            _server_started_by_playground = True

            for model, url, log_path in targets:
                status(f"Waiting for {model}'s server to respond…")
                if not await wait_for_health(url, log_path=log_path):
                    log_line(f"✗ {extract_error_summary(log_path)}")
                    return None

        for model, url, log_path in targets:
            status(f"Loading {model} (first run may download weights)…")
            if not await wait_for_model(url, model, timeout_s=load_timeout_s):
                log_line(f"✗ {extract_error_summary(log_path)}")
                return None

        return {model: url for model, url, _ in targets}
    finally:
        for tailer in tailers:
            await tailer.stop()
