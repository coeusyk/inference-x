from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from inference_x.engines.base import BaseEngine
from inference_x.utils.cuda_env import ensure_vllm_runtime_env
from inference_x.utils.vllm_pool_config import scale_model_config_for_pool
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)

logger = logging.getLogger(__name__)

_VLLM_AVAILABLE: bool | None = None


def _load_vllm() -> None:
    global _VLLM_AVAILABLE
    if _VLLM_AVAILABLE is not None:
        return
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

        globals()["LLM"] = LLM
        globals()["SamplingParams"] = SamplingParams
        _VLLM_AVAILABLE = True
    except ImportError:
        _VLLM_AVAILABLE = False


class VLLMEngine(BaseEngine):
    """vLLM-backed inference engine implementing the BaseEngine interface.

    Adapts the vLLM synchronous LLM API to the typed engine contract.
    The health check uses a stored flag rather than a live generation to avoid
    occupying the GPU unnecessarily on every /health poll.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        pool_size: int = 1,
        pool_models: list[str] | None = None,
        engine_index: int = 0,
        free_vram_gib: float | None = None,
    ) -> None:
        _load_vllm()
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed. "
                "Install it with: pip install vllm  (requires a CUDA-capable GPU)"
            )

        model_config = scale_model_config_for_pool(
            model_config,
            pool_size=pool_size,
            pool_models=pool_models,
            engine_index=engine_index,
            free_vram_gib=free_vram_gib,
        )
        required = {"name", "model_path"}
        missing = required - model_config.keys()
        if missing:
            raise ValueError(f"Missing required model config keys: {missing}")

        self._model_name: str = model_config["name"]
        self._model_path: str = model_config["model_path"]
        self._healthy = False
        self._engine_lock = threading.Lock()

        kwargs: dict[str, Any] = {
            "model": self._model_path,
            "dtype": "auto",
        }
        if "gpu_memory_utilization" in model_config:
            kwargs["gpu_memory_utilization"] = model_config["gpu_memory_utilization"]
        if "max_model_len" in model_config:
            kwargs["max_model_len"] = model_config["max_model_len"]
        if model_config.get("quantization"):
            kwargs["quantization"] = model_config["quantization"]
        if pool_size > 1:
            kwargs["enforce_eager"] = True

        if pool_size > 1 and "gpu_memory_utilization" in model_config:
            logger.info(
                "Multi-model pool (%d engines): gpu_memory_utilization=%s for model=%s",
                pool_size,
                model_config["gpu_memory_utilization"],
                model_config["name"],
            )

        logger.info("Initializing vLLM engine for model=%s path=%s", self._model_name, self._model_path)
        logger.info(
            "Loading weights (first run downloads from HuggingFace with no progress "
            "log until complete — can take several minutes on slow links)"
        )
        cuda_home = ensure_vllm_runtime_env(pool_size=pool_size)
        if cuda_home is None and not os.environ.get("CUDA_HOME"):
            logger.warning(
                "CUDA_HOME not set and nvcc not found; FlashInfer JIT may fail on WSL2"
            )
        LLM = globals()["LLM"]
        try:
            self._llm: LLM = LLM(**kwargs)
            self._supports_chat = self._detect_chat_support()
            self._healthy = True
            logger.info(
                "vLLM engine ready: model=%s chat_template=%s",
                self._model_name,
                self._supports_chat,
            )
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "OOM" in msg:
                raise RuntimeError(
                    f"CUDA OOM loading {self._model_path}. "
                    "Reduce gpu_memory_utilization or use a smaller model."
                ) from exc
            if "cache blocks" in msg.lower() or "kv cache" in msg.lower():
                raise RuntimeError(
                    f"GPU memory insufficient for KV cache loading {self._model_name}. "
                    "When running multiple models on one GPU, load fewer models or "
                    "lower gpu_memory_utilization / max_model_len in config/models.yaml."
                ) from exc
            if "not found" in msg.lower():
                raise RuntimeError(
                    f"Model not found: {self._model_path}. Check model_path in config/models.yaml."
                ) from exc
            if "nvcc" in msg.lower() or "cuda_home" in msg.lower():
                raise RuntimeError(
                    "vLLM FlashInfer JIT requires nvcc. On WSL2 without a system CUDA "
                    "toolkit, run via `uv run` so the bundled nvidia-cuda-nvcc wheel is "
                    "used, or set CUDA_HOME to your CUDA toolkit root."
                ) from exc
            raise RuntimeError(f"vLLM initialization failed: {msg}") from exc

    def _detect_chat_support(self) -> bool:
        try:
            tok = self._llm.get_tokenizer()
            return bool(getattr(tok, "chat_template", None))
        except Exception:
            return False

    @staticmethod
    def _messages_to_prompt(messages: list[ChatCompletionMessage | Any]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.role
            content = m.content
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _sampling_params(self, request: ChatCompletionRequest):
        SamplingParams = globals()["SamplingParams"]
        return SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.7,
            max_tokens=request.max_tokens if request.max_tokens is not None else 512,
            top_p=request.top_p if request.top_p is not None else 0.95,
        )

    def _stream_prompt(self, request: ChatCompletionRequest) -> str:
        if not self._supports_chat:
            return self._messages_to_prompt(request.messages)

        try:
            tokenizer = self._llm.get_tokenizer()
            vllm_messages = [
                {"role": m.role, "content": m.content} for m in request.messages
            ]
            return tokenizer.apply_chat_template(
                vllm_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return self._messages_to_prompt(request.messages)

    def _run_completion(self, request: ChatCompletionRequest):
        """Blocking completion using the startup-loaded sync engine."""
        sampling = self._sampling_params(request)
        vllm_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        with self._engine_lock:
            if self._supports_chat:
                return self._llm.chat(
                    messages=vllm_messages,  # type: ignore[arg-type]
                    sampling_params=sampling,
                    use_tqdm=False,
                )
            prompt = self._messages_to_prompt(request.messages)
            return self._llm.generate(
                prompts=[prompt],
                sampling_params=sampling,
                use_tqdm=False,
            )

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        if not _VLLM_AVAILABLE:
            yield "Streaming not available (vLLM not loaded)"
            return

        sync_queue: queue.Queue[str | None] = queue.Queue()

        def worker() -> None:
            sampling = self._sampling_params(request)
            prompt = self._stream_prompt(request)
            request_id = f"cmpl-stream-{uuid.uuid4().hex}"
            llm_engine = self._llm.llm_engine
            previous_text = ""

            try:
                with self._engine_lock:
                    llm_engine.add_request(request_id, prompt, sampling)
                    while llm_engine.has_unfinished_requests():
                        step_outputs = llm_engine.step()
                        if not step_outputs:
                            continue
                        for output in step_outputs:
                            if output.request_id != request_id or not output.outputs:
                                continue
                            text = output.outputs[0].text or ""
                            if text.startswith(previous_text):
                                chunk = text[len(previous_text) :]
                            else:
                                chunk = text
                            previous_text = text
                            if chunk:
                                sync_queue.put(chunk)
                            if output.finished:
                                return
            finally:
                sync_queue.put(None)

        try:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                chunk = await asyncio.to_thread(sync_queue.get)
                if chunk is None:
                    break
                yield chunk
            thread.join()
        except Exception as exc:
            raise RuntimeError(f"vLLM streaming generation failed: {exc}") from exc

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            outputs = await asyncio.to_thread(self._run_completion, request)
        except Exception as exc:
            raise RuntimeError(f"vLLM generation failed: {exc}") from exc

        out = outputs[0].outputs[0]
        generated = out.text
        finish = "stop" if out.finish_reason in ("stop", None) else "length"

        prompt_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        completion_tokens = len(out.token_ids) if out.token_ids else 0

        return ChatCompletionResponse(
            model=self._model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=generated),
                    finish_reason=finish,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def is_healthy(self) -> bool:
        return self._healthy
