"""llama.cpp backend: one external `llama-server` per engine (add-llama-cpp-backend).

The engine starts `llama-server` for the configured GGUF file, waits for it to
report healthy, and translates InferenceX's typed requests to its OpenAI-style
HTTP API. llama.cpp is never loaded in-process. See the change's design.md for
why each field below is reported the way it is (D1 to D7).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

import httpx

from inference_x.engines.base import BaseEngine, EngineUnavailableError
from inference_x.utils.llama_cpp_plan import BINARY_ENV, find_llama_server
from inference_x.routing.admission import ContextTooLongError
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatStreamChunk,
)

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_ENV = "INFERENCE_X_LLAMA_STARTUP_TIMEOUT_S"
_COMPLETION_TIMEOUT_S = 300.0
_PR_SET_PDEATHSIG = 1
_LOG_TAIL_LINES = 20


def _resolve_gguf(model_path: str, gguf_file: str | None) -> tuple[Path, str | None, str | None]:
    """(local file, hf repo, hf revision) for an entry, downloading into the
    Hugging Face cache on first use like vLLM entries do."""
    if gguf_file is None:
        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"GGUF file not found: {model_path}")
        return path, None, None
    from huggingface_hub import hf_hub_download

    try:
        local = Path(hf_hub_download(model_path, gguf_file))
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {gguf_file} from Hugging Face repo {model_path}: {exc}"
        ) from exc
    # The cache layout is <repo>/snapshots/<commit>/<file>: the directory name
    # is the revision the file was resolved from.
    return local, model_path, local.parent.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _free_port() -> int:
    # ponytail: bind-then-release has a small race; losing it makes llama-server
    # fail to bind and exit, which startup reports rather than misroutes.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _die_with_parent() -> None:
    """Runs in the child: have the kernel SIGTERM llama-server if InferenceX dies.

    The signal fires when the thread that spawned the child exits; engines are
    built on the startup thread, which lives as long as the process.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass


def _error_detail(response: httpx.Response) -> tuple[str | None, str]:
    """(llama-server error type, message) from an error response body."""
    try:
        error = response.json().get("error", {})
        return error.get("type"), str(error.get("message", response.text))
    except ValueError:
        return None, response.text


class LlamaCppEngine(BaseEngine):
    """Serves one GGUF model through an external `llama-server` process."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self._model_name: str = model_config["name"]
        self._model_path: str = model_config["model_path"]
        self._gguf_file: str | None = model_config.get("gguf_file")
        self._max_model_len: int = int(model_config["max_model_len"])
        self._instruction_tuned: bool = model_config.get("instruction_tuned", True)
        self._repetition_penalty: float | None = model_config.get("repetition_penalty")
        self._healthy = False
        self._process: subprocess.Popen[bytes] | None = None

        binary = find_llama_server()
        if binary is None:
            raise RuntimeError(
                f"llama-server not found: set {BINARY_ENV} to its path or put it on PATH"
            )
        gguf, self._hf_repo, self._hf_revision = _resolve_gguf(self._model_path, self._gguf_file)

        started = time.monotonic()
        self._weights_sha256 = _sha256(gguf)
        logger.info("GGUF sha256 %s (%.1fs)", self._weights_sha256, time.monotonic() - started)

        self._base_url = f"http://127.0.0.1:{_free_port()}"
        args = [
            binary, "-m", str(gguf), "-c", str(self._max_model_len), "-np", "1",
            "--host", "127.0.0.1", "--port", self._base_url.rsplit(":", 1)[1],
            "--no-webui", "-ctk", "f16", "-ctv", "f16",
            # Off, like the vLLM path's prefix caching: reusing a cached prefix
            # changes greedy output (measured, design D1), so a result would
            # depend on whatever request ran before it.
            "--no-cache-prompt",
        ]
        if model_config.get("n_gpu_layers") is not None:
            args += ["-ngl", str(model_config["n_gpu_layers"])]

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        self._log_path = logs_dir / f"llama-server-{self._model_name}.log"
        logger.info("Starting llama-server for model=%s (log: %s)", self._model_name, self._log_path)
        with self._log_path.open("wb") as log:
            self._process = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent
            )
        self._client = httpx.Client(base_url=self._base_url, timeout=10.0)
        try:
            self._wait_healthy()
            props = self._client.get("/props").json()
        except BaseException:
            self.shutdown()
            raise

        settings = props.get("default_generation_settings", {})
        self._n_ctx: int | None = settings.get("n_ctx")
        self._build_info: str | None = props.get("build_info")
        self._quantization: str | None = props.get("model_ftype")
        template = props.get("chat_template")
        self._chat_template_sha256 = (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if template else None
        )
        self._healthy = True
        logger.info(
            "llama-server ready: model=%s build=%s n_ctx=%s", self._model_name,
            self._build_info, self._n_ctx,
        )

    def _log_tail(self) -> str:
        try:
            lines = self._log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    def _wait_healthy(self) -> None:
        timeout_s = float(os.environ.get(_STARTUP_TIMEOUT_ENV, "300"))
        deadline = time.monotonic() + timeout_s
        assert self._process is not None
        while time.monotonic() < deadline:
            code = self._process.poll()
            if code is not None:
                raise RuntimeError(
                    f"llama-server exited with code {code} while loading "
                    f"{self._model_name}. Last log lines:\n{self._log_tail()}"
                )
            try:
                if self._client.get("/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        raise RuntimeError(
            f"llama-server did not become healthy within {timeout_s:.0f}s for "
            f"{self._model_name}. Last log lines:\n{self._log_tail()}"
        )

    # Provenance, read by manifest assembly (add-llama-cpp-backend D6).

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def kv_capacity_tokens(self) -> int | None:
        return self._n_ctx

    @property
    def chat_template_sha256(self) -> str | None:
        return self._chat_template_sha256

    @property
    def manifest_identity(self) -> dict[str, Any]:
        return {
            "backend": "llama.cpp",
            "backend_version": self._build_info,
            "hf_repo": self._hf_repo,
            "hf_revision": self._hf_revision,
            "gguf_file": self._gguf_file or Path(self._model_path).name,
            "weights_sha256": self._weights_sha256,
            "quantization": self._quantization,
            # torch's CUDA build describes vLLM's runtime, not the one
            # llama-server was compiled against.
            "hardware_cuda": None,
        }

    @property
    def runtime_snapshot(self) -> dict[str, Any]:
        return {
            "max_model_len": self._n_ctx,
            "kv_cache_dtype": "f16",
            "prefix_caching": False,
            "batch_invariant": False,
        }

    # Capabilities.

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int | None:
        # ponytail: a blocking localhost call from inside async admission
        # (a few ms, one slot per process); add an async variant if it shows up.
        #
        # Admission calls this before a streaming response starts, so a dead
        # server surfaces here as a 503 rather than as a 200 stream that breaks.
        if self._process is None or self._process.poll() is not None:
            raise self._unavailable(RuntimeError("process exited"))
        try:
            response = self._client.post(
                "/v1/chat/completions/input_tokens", json={"messages": self._messages(request)}
            )
            response.raise_for_status()
            return int(response.json()["input_tokens"])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("llama-server token count failed: %s", exc)
            return None

    def is_healthy(self) -> bool:
        return self._healthy and self._process is not None and self._process.poll() is None

    # Generation.

    @staticmethod
    def _messages(request: ChatCompletionRequest) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in request.messages]

    def _body(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": self._messages(request),
            "max_tokens": request.max_tokens if request.max_tokens is not None else 512,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 0.95,
            # Samplers InferenceX does not expose run disabled, so `resolved`
            # describes the whole sampler chain (D4). llama.cpp defaults are
            # top_k 40 and min_p 0.05.
            "top_k": 0,
            "min_p": 0.0,
            "stream": stream,
        }
        if not self._instruction_tuned:
            body["repeat_penalty"] = (
                self._repetition_penalty if self._repetition_penalty is not None else 1.15
            )
        if request.seed is not None:
            body["seed"] = request.seed
        if request.stop is not None:
            body["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _unavailable(self, exc: Exception) -> Exception:
        if self._process is None or self._process.poll() is not None:
            code = None if self._process is None else self._process.returncode
            self._healthy = False
            return EngineUnavailableError(
                f"llama-server for {self._model_name} is not running (exit code {code})"
            )
        return RuntimeError(f"llama-server request failed for {self._model_name}: {exc}")

    def _raise_for_error(self, response: httpx.Response) -> None:
        error_type, message = _error_detail(response)
        if error_type == "exceed_context_size_error":
            raise ContextTooLongError(message)
        raise RuntimeError(
            f"llama-server returned {response.status_code} for {self._model_name}: {message}"
        )

    @staticmethod
    def _usage(raw: dict[str, Any]) -> ChatCompletionUsage:
        return ChatCompletionUsage(
            prompt_tokens=raw["prompt_tokens"],
            completion_tokens=raw["completion_tokens"],
            total_tokens=raw["total_tokens"],
        )

    @staticmethod
    def _finish(value: Any) -> Literal["stop", "length"]:
        if value in ("stop", "length"):
            return value  # type: ignore[no-any-return]
        raise RuntimeError(f"llama-server returned unexpected finish_reason {value!r}")

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=_COMPLETION_TIMEOUT_S) as client:
                response = await client.post(
                    "/v1/chat/completions", json=self._body(request, stream=False)
                )
        except httpx.TransportError as exc:
            raise self._unavailable(exc) from exc
        if response.status_code != 200:
            self._raise_for_error(response)
        data = response.json()
        choice = data["choices"][0]
        # timing stays None: llama-server measures no queue time, and
        # EngineTiming is all-or-nothing (D6).
        return ChatCompletionResponse(
            model=self._model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=choice["message"].get("content") or ""),
                    finish_reason=self._finish(choice.get("finish_reason")),
                )
            ],
            usage=self._usage(data["usage"]),
        )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        finish: Literal["stop", "length"] | None = None
        usage: ChatCompletionUsage | None = None
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=None) as client:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=self._body(request, stream=True)
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._raise_for_error(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        if "error" in event:
                            raise RuntimeError(
                                f"llama-server stream error for {self._model_name}: {event['error']}"
                            )
                        if event.get("usage"):
                            usage = self._usage(event["usage"])
                        for choice in event.get("choices") or []:
                            content = (choice.get("delta") or {}).get("content")
                            if content:
                                yield ChatStreamChunk(content=content)
                            if choice.get("finish_reason") is not None:
                                finish = self._finish(choice["finish_reason"])
        except httpx.TransportError as exc:
            raise self._unavailable(exc) from exc
        if finish is None:
            raise RuntimeError(f"llama-server stream for {self._model_name} ended without a finish_reason")
        yield ChatStreamChunk(finish_reason=finish, usage=usage)

    def shutdown(self) -> None:
        self._healthy = False
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
        logger.info("llama-server stopped: model=%s", self._model_name)
