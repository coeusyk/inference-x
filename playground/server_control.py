"""Start/stop the playground inference server with selected models loaded."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_PATH = _REPO_ROOT / "logs" / "playground-server.log"

_server_started_by_playground: bool = False


def playground_started_server() -> bool:
    """True when this process started the background uvicorn server."""
    return _server_started_by_playground


def _reset_playground_server_state() -> None:
    """Clear the started-server flag (for tests)."""
    global _server_started_by_playground
    _server_started_by_playground = False


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
    models: list[str],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Launch uvicorn with ``INFERENCE_X_LOADED_MODELS`` set to *models*."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["INFERENCE_X_LOADED_MODELS"] = ",".join(models)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    log_f = _LOG_PATH.open("a", encoding="utf-8")
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
) -> bool:
    """Start or restart the server so *models* are loaded and ready for inference."""
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
        return False

    def status(message: str) -> None:
        if on_status is not None:
            on_status(message)

    def log_line(message: str) -> None:
        if on_log is not None:
            on_log(message)

    tailer = LogTailer(log_line, log_path=_LOG_PATH)
    prepare_log_session(_LOG_PATH)
    tailer.start(from_offset=0)

    try:
        loaded = await fetch_loaded_models(base_url)
        if set(wanted) <= set(loaded):
            for name in wanted:
                status(f"Waiting for {name} to load…")
                if not await wait_for_model(base_url, name, timeout_s=load_timeout_s):
                    log_line(f"✗ Timed out waiting for {name}")
                    return False
            return True

        status(f"Starting server with {', '.join(wanted)}…")
        await stop_playground_server()
        await start_playground_server(wanted)
        _server_started_by_playground = True

        status("Waiting for server to respond…")
        if not await wait_for_health(base_url, log_path=_LOG_PATH):
            log_line(f"✗ {extract_error_summary(_LOG_PATH)}")
            return False

        for name in wanted:
            status(f"Loading {name} (first run may download weights)…")
            if not await wait_for_model(base_url, name, timeout_s=load_timeout_s):
                log_line(f"✗ {extract_error_summary(_LOG_PATH)}")
                return False
        return True
    finally:
        await tailer.stop()
