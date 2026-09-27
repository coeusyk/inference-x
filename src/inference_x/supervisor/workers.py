"""Worker processes and the lifecycle state machine (add-model-lifecycle-supervisor).

A worker is an unchanged InferenceX app serving one model (DEC-059). The
supervisor starts workers on demand, keeps at most `max_loaded` running,
evicts the least recently used idle one to make room, and never stops a
worker with requests in flight. See the change's design.md, D1 to D5.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from inference_x.utils.process import die_with_parent, free_port

logger = logging.getLogger(__name__)

_LOG_TAIL_LINES = 20
_STOP_GRACE_S = 30.0


class ModelLoadFailed(RuntimeError):
    """A worker could not be started for the model (503 `model_load_failed`)."""


class ModelBusy(RuntimeError):
    """No worker could be freed within the switch wait (503 `model_busy`)."""


def default_worker_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "uvicorn", "inference_x.api.main:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]


class Worker:
    """One running worker process and the bookkeeping the supervisor needs."""

    def __init__(self, model: str, command: Callable[[int], list[str]], logs_dir: Path) -> None:
        self.model = model
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = logs_dir / f"worker-{model}.log"
        self.started_at = time.time()
        self.last_used = self.started_at
        self.in_flight = 0
        self.stopping = False
        env = dict(os.environ)
        env["INFERENCE_X_DEFAULT_MODEL"] = model
        env.pop("INFERENCE_X_LOADED_MODELS", None)
        with self.log_path.open("wb") as log:
            self.process = subprocess.Popen(
                command(self.port), stdout=log, stderr=subprocess.STDOUT, env=env,
                preexec_fn=die_with_parent,
            )

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.poll() is None

    def log_tail(self) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    async def wait_healthy(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if not self.alive():
                    raise ModelLoadFailed(
                        f"worker for {self.model} exited with code {self.process.returncode} "
                        f"while loading. Last log lines:\n{self.log_tail()}"
                    )
                try:
                    if (await client.get(f"{self.url}/health")).status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.25)
        raise ModelLoadFailed(
            f"worker for {self.model} was not healthy after {timeout_s:.0f}s. "
            f"Last log lines:\n{self.log_tail()}"
        )

    async def stop(self) -> None:
        """SIGTERM (uvicorn runs the app's shutdown, which stops the engine),
        then SIGKILL if it hasn't exited within the grace period."""
        self.stopping = True
        if self.alive():
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, _STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        logger.info("worker stopped: model=%s exit=%s", self.model, self.process.returncode)


class Supervisor:
    def __init__(
        self,
        *,
        max_loaded: int = 1,
        switch_wait_s: float = 60.0,
        startup_timeout_s: float = 600.0,
        worker_command: Callable[[int], list[str]] = default_worker_command,
        logs_dir: Path = Path("logs"),
    ) -> None:
        self._max_loaded = max_loaded
        self._switch_wait_s = switch_wait_s
        self._startup_timeout_s = startup_timeout_s
        self._command = worker_command
        self._logs_dir = logs_dir
        logs_dir.mkdir(exist_ok=True)
        self._workers: dict[str, Worker] = {}
        self._loading: dict[str, asyncio.Future[None]] = {}
        self._failures: dict[str, dict[str, Any]] = {}
        # One GPU: loads, evictions and unloads happen one at a time.
        self._lock = asyncio.Lock()

    # Request path -------------------------------------------------------------

    async def acquire(self, model: str) -> Worker:
        """A running worker for *model* with this request counted in flight,
        loading the model first if needed. Pair every call with `release`."""
        while True:
            worker = self._running(model)
            if worker is not None:
                worker.in_flight += 1
                worker.last_used = time.time()
                return worker
            load = self._loading.get(model)
            if load is None:
                load = asyncio.ensure_future(self._load(model))
                self._loading[model] = load

                def _done(_: asyncio.Future[None], m: str = model) -> None:
                    self._loading.pop(m, None)

                load.add_done_callback(_done)
            await asyncio.shield(load)

    def release(self, worker: Worker) -> None:
        worker.in_flight -= 1
        worker.last_used = time.time()

    def reap(self, model: str) -> None:
        """Forget a worker that has exited on its own (D5)."""
        worker = self._workers.get(model)
        if worker is not None and not worker.alive() and not worker.stopping:
            self._workers.pop(model)
            self._failures[model] = {
                "exit_code": worker.process.returncode,
                "reason": f"worker exited with code {worker.process.returncode}",
                "log_tail": worker.log_tail(),
                "at": time.time(),
            }
            logger.error("worker for %s exited with code %s", model, worker.process.returncode)

    def _running(self, model: str) -> Worker | None:
        self.reap(model)
        worker = self._workers.get(model)
        return worker if worker is not None and not worker.stopping else None

    # Lifecycle ------------------------------------------------------------------

    async def _load(self, model: str) -> None:
        async with self._lock:
            if self._running(model) is not None:
                return
            await self._make_room(exclude=model)
            logger.info("loading worker for model=%s", model)
            worker = Worker(model, self._command, self._logs_dir)
            try:
                await worker.wait_healthy(self._startup_timeout_s)
            except ModelLoadFailed as exc:
                await worker.stop()
                self._failures[model] = {
                    "exit_code": worker.process.returncode,
                    "reason": str(exc).split("\n", 1)[0],
                    "log_tail": worker.log_tail(),
                    "at": time.time(),
                }
                raise
            self._failures.pop(model, None)
            self._workers[model] = worker
            logger.info("worker ready: model=%s pid=%s port=%s", model, worker.pid, worker.port)

    async def _make_room(self, exclude: str) -> None:
        """Stop least recently used idle workers until a slot is free (D4)."""
        deadline = time.monotonic() + self._switch_wait_s
        while True:
            for name in list(self._workers):
                self.reap(name)
            others = [w for n, w in self._workers.items() if n != exclude and not w.stopping]
            if len(others) < self._max_loaded:
                return
            idle = [w for w in others if w.in_flight == 0]
            if idle:
                victim = min(idle, key=lambda w: w.last_used)
                # No await between the in_flight check and this flag, so no
                # request can start on the victim once it is chosen.
                victim.stopping = True
                logger.info("evicting model=%s to load %s", victim.model, exclude)
                await victim.stop()
                self._workers.pop(victim.model, None)
                continue
            if time.monotonic() >= deadline:
                raise ModelBusy(
                    f"cannot load {exclude}: every loaded model has requests in flight "
                    f"after {self._switch_wait_s:.0f}s"
                )
            await asyncio.sleep(0.1)

    async def load(self, model: str) -> None:
        worker = await self.acquire(model)
        self.release(worker)

    async def unload(self, model: str) -> bool:
        """Stop *model*'s worker once idle. False when it was not running;
        ModelBusy when it stayed busy for the whole switch wait."""
        async with self._lock:
            worker = self._running(model)
            if worker is None:
                return False
            deadline = time.monotonic() + self._switch_wait_s
            while worker.in_flight > 0:
                if time.monotonic() >= deadline:
                    raise ModelBusy(f"{model} still has {worker.in_flight} request(s) in flight")
                await asyncio.sleep(0.1)
            worker.stopping = True
            await worker.stop()
            self._workers.pop(model, None)
            return True

    async def shutdown(self) -> None:
        for worker in list(self._workers.values()):
            await worker.stop()
        self._workers.clear()

    # Reporting ------------------------------------------------------------------

    def state(self, model: str) -> dict[str, Any]:
        worker = self._running(model)
        if worker is not None:
            return {
                "model": model, "state": "loaded", "pid": worker.pid, "port": worker.port,
                "started_at": worker.started_at, "last_used": worker.last_used,
                "in_flight": worker.in_flight,
            }
        if model in self._loading:
            return {"model": model, "state": "loading"}
        failure = self._failures.get(model)
        if failure is not None:
            return {"model": model, "state": "failed", **failure}
        return {"model": model, "state": "unloaded"}

    def loaded(self) -> list[Worker]:
        return [w for n in list(self._workers) if (w := self._running(n)) is not None]
