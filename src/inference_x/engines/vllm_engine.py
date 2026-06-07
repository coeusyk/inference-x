from __future__ import annotations

import logging
from typing import Any

from inference_x.engines.base import BaseEngine
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMEngine(BaseEngine):
    """vLLM-backed inference engine implementing the BaseEngine interface.

    Adapts the vLLM synchronous LLM API to the typed engine contract.
    The health check uses a stored flag rather than a live generation to avoid
    occupying the GPU unnecessarily on every /health poll.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed. "
                "Install it with: pip install vllm  (requires a CUDA-capable GPU)"
            )

        required = {"name", "model_path"}
        missing = required - model_config.keys()
        if missing:
            raise ValueError(f"Missing required model config keys: {missing}")

        self._model_name: str = model_config["name"]
        self._model_path: str = model_config["model_path"]
        self._healthy = False

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

        logger.info("Initializing vLLM engine for model=%s path=%s", self._model_name, self._model_path)
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
            if "not found" in msg.lower():
                raise RuntimeError(
                    f"Model not found: {self._model_path}. Check model_path in config/models.yaml."
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

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        sampling = SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.7,
            max_tokens=request.max_tokens if request.max_tokens is not None else 512,
            top_p=request.top_p if request.top_p is not None else 0.95,
        )

        vllm_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            if self._supports_chat:
                outputs = self._llm.chat(
                    messages=vllm_messages,  # type: ignore[arg-type]
                    sampling_params=sampling,
                    use_tqdm=False,
                )
            else:
                prompt = self._messages_to_prompt(request.messages)
                outputs = self._llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling,
                    use_tqdm=False,
                )
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
