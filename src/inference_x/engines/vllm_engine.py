from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from inference_x.engines.base import BaseEngine
from inference_x.utils.cuda_env import ensure_vllm_runtime_env
from inference_x.utils.vllm_pool_config import scale_model_config_for_pool
from inference_x.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatStreamChunk,
)

logger = logging.getLogger(__name__)

_VLLM_AVAILABLE: bool | None = None
# Safety net against a request that never reaches a finished RequestOutput.
# Unlike the EngineDriver this replaced, a timeout here triggers a real
# engine-side abort (AsyncLLM.generate() reacts to CancelledError by calling
# EngineCore.abort_requests_async) rather than merely giving up on waiting —
# see migrate-async-llm-engine design.md Decision 7. Scope unchanged from
# before the migration: this bounds only the non-streaming generate() path
# (via its own asyncio.timeout wrapper below), not generate_stream().
_COMPLETION_TIMEOUT_S = 300.0


def derive_terminal_metadata(
    output: Any,
) -> tuple[Literal["stop", "length"], ChatCompletionUsage]:
    """Map a finished vLLM `RequestOutput` to a finish reason and engine usage.

    Shared by the streaming terminal chunk and non-streaming `generate`, so the
    two paths cannot drift: OS-2 requires that a streamed and a non-streamed
    request for the same prompt report the same `completion_tokens`, and one
    function is the only way to guarantee that rather than hope for it.

    Counts come from vLLM's own token ids. Nothing here counts text.

    Relocated verbatim from the deleted `engines/driver.py` during the
    migrate-async-llm-engine change (DEC-050 §3 / OWN-B5): moved, not
    reimplemented — the body is unchanged from its EngineDriver-era form.
    """
    completion = output.outputs[0]
    finish: Literal["stop", "length"] = (
        "stop" if completion.finish_reason in ("stop", None) else "length"
    )
    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
    completion_tokens = len(completion.token_ids) if completion.token_ids else 0
    return finish, ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _probe_cuda_vram() -> dict[str, float | bool]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free, total = torch.cuda.mem_get_info(0)
        return {
            "cuda_available": True,
            "free_gib": round(free / (1024**3), 2),
            "total_gib": round(total / (1024**3), 2),
        }
    except Exception as exc:
        return {"cuda_available": False, "probe_error": str(exc)}


def _vram_budget_error(
    model_name: str,
    gpu_util: float,
    free_gib: float,
    total_gib: float,
) -> RuntimeError:
    requested = gpu_util * total_gib
    return RuntimeError(
        f"Insufficient GPU memory to start {model_name}. "
        f"Free VRAM {free_gib:.2f} GiB is less than "
        f"gpu_memory_utilization={gpu_util} requires ({requested:.2f} GiB of "
        f"{total_gib:.2f} GiB total). Lower gpu_memory_utilization in "
        "config/models.yaml, stop other GPU processes, or use a smaller model."
    )


def _check_vram_budget(
    model_name: str,
    gpu_util: float | None,
    vram: dict[str, float | bool],
) -> None:
    if gpu_util is None or not vram.get("cuda_available"):
        return
    free = vram.get("free_gib")
    total = vram.get("total_gib")
    if not isinstance(free, (int, float)) or not isinstance(total, (int, float)):
        return
    requested = float(gpu_util) * float(total)
    if float(free) < requested:
        raise _vram_budget_error(model_name, float(gpu_util), float(free), float(total))


_GATED_REPO_HINT = (
    "Request access on HuggingFace, then authenticate:\n"
    "  1. Visit the model page and accept the license\n"
    "  2. uv run hf auth login\n"
    "  3. Or export HF_TOKEN=<your-token> before starting the server"
)


def resolve_hf_token() -> str | None:
    """Return a HuggingFace token from the environment, if set."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _is_hf_hub_model_path(model_path: str) -> bool:
    """True when *model_path* is a HuggingFace Hub repo id (org/name), not a local path."""
    if not model_path:
        return False
    if model_path.startswith(("/", "./", "../", "~")):
        return False
    if os.path.isabs(model_path):
        return False
    # Windows absolute paths: C:\... or C:/...
    if len(model_path) >= 2 and model_path[1] == ":":
        return False
    parts = model_path.split("/")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


def preflight_hf_access(
    model_name: str,
    model_path: str,
    token: str | None,
) -> None:
    """Verify HuggingFace repo access before vLLM starts weight download.

    Skips local filesystem paths. For Hub repos that are gated or private, probes
    a lightweight config.json fetch so both missing-token and not-yet-approved
    cases fail fast with a short message.
    """
    if not _is_hf_hub_model_path(model_path):
        return

    try:
        from huggingface_hub import hf_hub_download, model_info
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        logger.warning(
            "huggingface_hub not available; skipping HF preflight for %s",
            model_path,
        )
        return

    try:
        info = model_info(model_path, token=token)
    except RepositoryNotFoundError:
        raise RuntimeError(
            f"Model '{model_path}' not found on HuggingFace. "
            "Check model_path in config/models.yaml."
        ) from None

    gated = bool(getattr(info, "gated", False))
    private = bool(getattr(info, "private", False))
    if not gated and not private:
        return

    try:
        hf_hub_download(repo_id=model_path, filename="config.json", token=token)
    except GatedRepoError:
        if token:
            raise RuntimeError(
                f"Access to '{model_path}' is not yet approved for your HuggingFace "
                f"account. Request access at https://huggingface.co/{model_path} "
                "and wait for approval."
            ) from None
        raise RuntimeError(
            f"Model '{model_name}' ({model_path}) is gated on HuggingFace.\n"
            f"{_GATED_REPO_HINT}"
        ) from None
    except RepositoryNotFoundError:
        raise RuntimeError(
            f"Model '{model_path}' not found on HuggingFace. "
            "Check model_path in config/models.yaml."
        ) from None


def _gated_model_startup_error(model_name: str, model_path: str) -> RuntimeError:
    return RuntimeError(
        f"Model '{model_name}' ({model_path}) is gated on HuggingFace.\n"
        f"{_GATED_REPO_HINT}"
    )


def _map_vllm_init_error(model_name: str, model_path: str, exc: Exception) -> RuntimeError:
    """Turn low-level vLLM/HuggingFace failures into actionable RuntimeErrors."""
    msg = str(exc)
    lower = msg.lower()
    if (
        "gated repo" in lower
        or "gatedrepo" in lower
        or "not in the authorized list" in lower
        or ("403" in msg and "forbidden" in lower)
    ):
        return _gated_model_startup_error(model_name, model_path)
    if "out of memory" in lower or "oom" in msg:
        return RuntimeError(
            f"CUDA OOM loading {model_path}. "
            "Reduce gpu_memory_utilization or use a smaller model."
        )
    if (
        "free memory on device" in lower
        or ("gpu memory utilization" in lower and "less than desired" in lower)
    ):
        return RuntimeError(
            f"GPU memory insufficient to start {model_name}. "
            "Lower gpu_memory_utilization in config/models.yaml, stop other GPU "
            "processes (e.g. a previous ./scripts/dev.sh serve), or use a smaller model."
        )
    if "cache blocks" in lower or "kv cache" in lower or "no available memory for the cache" in lower:
        return RuntimeError(
            f"GPU memory insufficient for KV cache loading {model_name}. "
            "On ≤6 GiB GPUs, load one model: "
            "INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b ./scripts/dev.sh serve. "
            "For a single model, raise gpu_memory_utilization (e.g. 0.85) and/or "
            "lower max_model_len (e.g. 2048) in config/models.yaml. "
            "Stop other GPU processes (nvidia-smi) before retrying."
        )
    if "mamba cache blocks" in lower or "max_num_seqs" in lower:
        return RuntimeError(
            f"GPU memory insufficient for Mamba/state cache loading {model_name}. "
            "Hybrid models (e.g. Qwen3.5) need more VRAM headroom than dense models. "
            "Set max_num_seqs (e.g. 64) and/or lower max_model_len in config/models.yaml, "
            "or raise gpu_memory_utilization."
        )
    if "not found" in lower:
        return RuntimeError(
            f"Model not found: {model_path}. Check model_path in config/models.yaml."
        )
    if "greater than the derived max_model_len" in lower:
        limit_match = re.search(r"derived max_model_len \(max_position_embeddings=([\d.]+)", msg)
        limit_hint = limit_match.group(1).rstrip(".0") if limit_match else "the model limit"
        return RuntimeError(
            f"max_model_len in config/models.yaml for {model_name} exceeds the "
            f"model's position limit ({limit_hint}). Lower max_model_len for "
            f"{model_name} in config/models.yaml."
        )
    if "nvcc" in lower or "cuda_home" in lower:
        return RuntimeError(
            "vLLM FlashInfer JIT requires nvcc. On WSL2 without a system CUDA "
            "toolkit, run via `uv run` so the bundled nvidia-cuda-nvcc wheel is "
            "used, or set CUDA_HOME to your CUDA toolkit root."
        )
    if "engine core initialization failed" in lower:
        vram = _probe_cuda_vram()
        total = vram.get("total_gib")
        if isinstance(total, (int, float)) and float(total) <= 10 and "8b" in model_name.lower():
            return RuntimeError(
                f"vLLM failed loading {model_name} on a {float(total):.0f} GiB GPU. "
                "Llama 3 8B in bf16 needs roughly 16 GiB VRAM for weights alone. "
                "Use a smaller model (e.g. qwen2.5-1.5b) or a quantized checkpoint."
            )
        if isinstance(total, (int, float)) and vram.get("free_gib") is not None:
            free = float(vram["free_gib"])  # type: ignore[arg-type]
            return RuntimeError(
                f"vLLM engine subprocess failed for {model_name}. "
                f"GPU has {free:.2f}/{float(total):.2f} GiB free. "
                "Lower gpu_memory_utilization in config/models.yaml, stop other "
                "GPU processes, or use a smaller model."
            )
    return RuntimeError(f"vLLM initialization failed for {model_name}: {msg}")


def _load_vllm() -> None:
    global _VLLM_AVAILABLE
    if _VLLM_AVAILABLE is not None:
        return
    try:
        from vllm import SamplingParams  # type: ignore[import-untyped]
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore[import-untyped]
        from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore[import-untyped]

        globals()["AsyncLLM"] = AsyncLLM
        globals()["AsyncEngineArgs"] = AsyncEngineArgs
        globals()["SamplingParams"] = SamplingParams
        _VLLM_AVAILABLE = True
    except ImportError:
        _VLLM_AVAILABLE = False


class VLLMEngine(BaseEngine):
    """vLLM-backed inference engine implementing the BaseEngine interface.

    Adapts vLLM's async engine (`AsyncLLM`) to the typed engine contract.
    The health check reads `AsyncLLM.errored` rather than running a live
    generation, avoiding GPU work on every /health poll.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        pool_size: int = 1,
        pool_models: list[str] | None = None,
        pool_configs: list[dict[str, Any]] | None = None,
        engine_index: int = 0,
        free_vram_gib: float | None = None,
        total_vram_gib: float | None = None,
        session_free_vram_gib: float | None = None,
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
            pool_configs=pool_configs,
            engine_index=engine_index,
            free_vram_gib=free_vram_gib,
            total_vram_gib=total_vram_gib or _probe_cuda_vram().get("total_gib") or 8.0,
            session_free_vram_gib=session_free_vram_gib,
        )
        required = {"name", "model_path"}
        missing = required - model_config.keys()
        if missing:
            raise ValueError(f"Missing required model config keys: {missing}")

        self._model_name: str = model_config["name"]
        self._model_path: str = model_config["model_path"]
        self._pool_size = pool_size
        self._max_completion_tokens: int | None = model_config.get("max_completion_tokens")
        self._instruction_tuned: bool = bool(model_config.get("instruction_tuned", True))
        self._repetition_penalty: float | None = model_config.get("repetition_penalty")
        self._healthy = False
        self._kv_capacity_tokens: int | None = None

        hf_token = resolve_hf_token()
        preflight_hf_access(self._model_name, self._model_path, hf_token)

        kwargs: dict[str, Any] = {
            "model": self._model_path,
            "dtype": "auto",
            "trust_remote_code": True,
        }
        if hf_token:
            kwargs["hf_token"] = hf_token
        if "gpu_memory_utilization" in model_config:
            kwargs["gpu_memory_utilization"] = model_config["gpu_memory_utilization"]
        max_model_len = model_config.get("max_model_len")
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        max_num_seqs = model_config.get("max_num_seqs")
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs
        max_num_batched_tokens = model_config.get("max_num_batched_tokens")
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        block_size = model_config.get("block_size")
        if block_size is not None:
            kwargs["block_size"] = block_size
        kv_cache_dtype = model_config.get("kv_cache_dtype")
        if kv_cache_dtype is not None:
            kwargs["kv_cache_dtype"] = kv_cache_dtype
        if model_config.get("enable_prefix_caching") is not None:
            kwargs["enable_prefix_caching"] = model_config["enable_prefix_caching"]
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
        AsyncEngineArgs = globals()["AsyncEngineArgs"]
        AsyncLLM = globals()["AsyncLLM"]
        gpu_util = kwargs.get("gpu_memory_utilization")
        if isinstance(gpu_util, (int, float)):
            _check_vram_budget(self._model_name, float(gpu_util), _probe_cuda_vram())
        try:
            engine_args = AsyncEngineArgs(**kwargs)
            self._llm: AsyncLLM = AsyncLLM.from_engine_args(engine_args)
            self._supports_chat = self._detect_chat_support()
            self._healthy = True
            self._log_kv_cache_stats()
            logger.info(
                "vLLM engine ready: model=%s chat_template=%s",
                self._model_name,
                self._supports_chat,
            )
        except Exception as exc:
            raise _map_vllm_init_error(self._model_name, self._model_path, exc) from exc

    def _log_kv_cache_stats(self) -> None:
        """Log vLLM KV-cache sizing after engine init (mirrors vLLM '# GPU blocks' lines).

        Also records ``self._kv_capacity_tokens`` so the /v1/metrics route can report
        real KV-pool capacity instead of the pre-load estimate in vllm_pool_config.
        """
        try:
            # AsyncLLM exposes vllm_config directly as an instance attribute (no
            # llm_engine indirection, unlike the offline LLM class this replaced) —
            # verified against vLLM 0.22.1's AsyncLLM/EngineCoreClient
            # (migrate-async-llm-engine Decision 6).
            cache_config = getattr(
                getattr(self._llm, "vllm_config", None), "cache_config", None
            )
            num_blocks = getattr(cache_config, "num_gpu_blocks", None)
            block_size = getattr(cache_config, "block_size", None)
            if num_blocks is not None and block_size is not None:
                self._kv_capacity_tokens = int(num_blocks) * int(block_size)
                logger.info(
                    "KV cache for model=%s: %d GPU blocks × %d tokens/block "
                    "(~%d tokens capacity)",
                    self._model_name,
                    num_blocks,
                    block_size,
                    self._kv_capacity_tokens,
                )
        except Exception as exc:
            logger.debug("KV cache stats unavailable for %s: %s", self._model_name, exc)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def kv_capacity_tokens(self) -> int | None:
        """Real post-load KV-cache capacity in tokens (num_gpu_blocks * block_size).

        ``None`` if unavailable (e.g. vLLM not loaded, or cache_config introspection
        failed — see _log_kv_cache_stats).
        """
        return getattr(self, "_kv_capacity_tokens", None)

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

    def count_prompt_tokens(self, request: ChatCompletionRequest) -> int:
        """Real prompt token count via this model's tokenizer, for admission control.

        Overrides the BaseEngine capability declared per DEC-047 §3, which
        routing/admission.py now calls directly rather than discovering with
        getattr. Narrowing the declared ``int | None`` to ``int`` is deliberate:
        this engine always produces a number, falling back to a chars/4 estimate
        — the same heuristic AdmissionController uses for engines with no
        tokenizer at all — if tokenization fails for any reason.

        A consequence worth naming: because this returns an estimate rather than
        None on tokenizer failure, admission cannot tell that path from a real
        count and emits no `prompt_tokens_estimated` warning for it. Recorded as a
        known limitation of OS-4 rather than changed here — the engine's own
        fallback contract is out of that change's scope.
        """
        try:
            prompt = self._stream_prompt(request)
            tokenizer = self._llm.get_tokenizer()
            return len(tokenizer.encode(prompt))
        except Exception as exc:
            logger.debug(
                "count_prompt_tokens fallback for model=%s: %s", self._model_name, exc
            )
            total_chars = sum(len(m.content) for m in request.messages)
            return max(1, total_chars // 4)

    def _resolve_max_tokens(self, request: ChatCompletionRequest) -> int:
        if self._max_completion_tokens is not None:
            return self._max_completion_tokens
        return request.max_tokens if request.max_tokens is not None else 512

    def _sampling_params(self, request: ChatCompletionRequest):
        SamplingParams = globals()["SamplingParams"]
        kwargs: dict[str, Any] = {
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_tokens": self._resolve_max_tokens(request),
            "top_p": request.top_p if request.top_p is not None else 0.95,
        }
        if not self._instruction_tuned:
            kwargs["repetition_penalty"] = (
                self._repetition_penalty if self._repetition_penalty is not None else 1.15
            )
        # DEC-051 / OS-3: forward seed unchanged when set; omit when None.
        # Do not normalize backend sentinels (e.g. -1).
        if request.seed is not None:
            kwargs["seed"] = request.seed
        return SamplingParams(**kwargs)

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

    async def _stream_chunks(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Single call site into `AsyncLLM.generate()` (migrate-async-llm-engine
        Decision 2). Both `generate_stream` and `generate` are expressed in
        terms of this — there is only one place that talks to the engine, and
        only one place `derive_terminal_metadata` is called, so the two public
        methods cannot report different counts (DEC-050).

        Deliberately carries no timeout: `generate_stream` and `generate` apply
        timeout policy differently (Decision 7 / Task 2.7 — `generate` wraps
        its own consumption of this generator; `generate_stream` does not,
        preserving the pre-migration policy where only the non-streaming path
        was time-bounded).

        A disconnected/cancelled consumer propagates `GeneratorExit`/
        `CancelledError` straight into `AsyncLLM.generate()`'s own consumption
        loop, which reacts by aborting the request on the engine core
        (Decision 4) — this must not be caught and swallowed here.
        """
        sampling = self._sampling_params(request)
        prompt = self._stream_prompt(request)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        previous_text = ""
        try:
            async for output in self._llm.generate(prompt, sampling, request_id):
                if not output.outputs:
                    continue
                text = output.outputs[0].text or ""
                chunk = text[len(previous_text) :] if text.startswith(previous_text) else text
                previous_text = text
                if chunk:
                    yield ChatStreamChunk(content=chunk)
                if output.finished:
                    finish, usage = derive_terminal_metadata(output)
                    yield ChatStreamChunk(content="", finish_reason=finish, usage=usage)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            raise RuntimeError(f"vLLM streaming generation failed: {exc}") from exc

    async def generate_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Yield engine chunks (DEC-049). See `_stream_chunks` for the shared
        engine-consumption primitive both this and `generate` use.
        """
        if not _VLLM_AVAILABLE:
            # Pre-OS-2 wire behavior: a single content message, no terminal
            # finish_reason event (OpenSpec R3 — error path unchanged).
            yield ChatStreamChunk(
                content="Streaming not available (vLLM not loaded)",
            )
            return

        async for chunk in self._stream_chunks(request):
            yield chunk

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM generation failed: vLLM not loaded")

        generated_parts: list[str] = []
        finish: Literal["stop", "length"] = "stop"
        usage: ChatCompletionUsage | None = None
        try:
            async with asyncio.timeout(_COMPLETION_TIMEOUT_S):
                async for chunk in self._stream_chunks(request):
                    if chunk.content:
                        generated_parts.append(chunk.content)
                    if chunk.finish_reason is not None:
                        # _stream_chunks only ever sets finish_reason via
                        # derive_terminal_metadata, which returns "stop"/"length"
                        # — never "error" (that value is only used elsewhere, on
                        # the SSE error path). Narrow explicitly rather than
                        # widen `finish`'s type to match ChatStreamChunk's.
                        finish = "length" if chunk.finish_reason == "length" else "stop"
                        usage = chunk.usage
        except TimeoutError as exc:
            raise RuntimeError(
                f"vLLM completion timed out after {_COMPLETION_TIMEOUT_S}s "
                f"for model={self._model_name}"
            ) from exc

        assert usage is not None, "engine finished without terminal usage metadata"
        return ChatCompletionResponse(
            model=self._model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content="".join(generated_parts)),
                    finish_reason=finish,
                )
            ],
            usage=usage,
        )

    def is_healthy(self) -> bool:
        llm = getattr(self, "_llm", None)
        if llm is not None and llm.errored:
            return False
        return self._healthy

    def shutdown(self) -> None:
        """Stop the vLLM engine subprocess and release multiprocessing resources."""
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        self._healthy = False
        try:
            llm.shutdown()
        except Exception as exc:
            logger.warning(
                "Error shutting down vLLM engine for %s: %s", self._model_name, exc
            )
        finally:
            self._llm = None  # type: ignore[assignment]
            logger.info("vLLM engine shut down: model=%s", self._model_name)
