import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any

from vllm import LLM, SamplingParams

from .schemas import CompletionRequest


logger = logging.getLogger(__name__)


class LLMEngine(ABC):
    """Base interface - will never change"""
    
    @abstractmethod
    async def generate(self, request: CompletionRequest) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        pass


# Phase 1: Single vLLM implementation
class VLLMEngine(LLMEngine):
    """vLLM-based LLM engine with OpenAI-compatible API."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize vLLM engine with model configuration.
        
        Args:
            model_config: Dictionary containing model configuration with keys:
                - name: Model identifier
                - model_path: HuggingFace model path or local path
                - gpu_memory_utilization: GPU memory fraction (0.0-1.0)
                - max_model_len: Maximum sequence length
                - quantization: Quantization method (optional)
                
        Raises:
            ValueError: If required config keys are missing
            RuntimeError: If vLLM initialization fails (CUDA OOM, model not found)
        """

        logger.info(f"Initializing VLLMEngine with config: {model_config}")
        
        # Validate required config keys
        required_keys = ["name", "model_path"]
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        self.model_name = model_config["name"]
        self.model_path = model_config["model_path"]
        
        try:
            # Initialize vLLM with configuration
            llm_kwargs = {"model": self.model_path}
            
            # Add optional parameters if provided
            if "gpu_memory_utilization" in model_config:
                llm_kwargs["gpu_memory_utilization"] = model_config["gpu_memory_utilization"]
            
            if "max_model_len" in model_config:
                llm_kwargs["max_model_len"] = model_config["max_model_len"]
            
            if model_config.get("quantization"):
                llm_kwargs["quantization"] = model_config["quantization"]
            
            # Set dtype to auto for optimal performance
            llm_kwargs["dtype"] = "auto"
            
            logger.info(f"Initializing vLLM with parameters: {llm_kwargs}")
            self.llm = LLM(**llm_kwargs)
            
            # Detect whether the model supports chat templates
            self._supports_chat = self._check_chat_support()
            logger.info(
                f"Successfully initialized VLLMEngine for model: {self.model_name} "
                f"(chat_support={self._supports_chat})"
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize vLLM engine: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Provide more specific error messages for common issues
            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                raise RuntimeError(
                    f"CUDA out of memory when loading model. "
                    f"Try reducing gpu_memory_utilization or using a smaller model."
                ) from e
            elif "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model not found at path: {self.model_path}. "
                    f"Verify the model path is correct."
                ) from e
            else:
                raise RuntimeError(error_msg) from e
    
    def _check_chat_support(self) -> bool:
        """Check if the loaded model has a chat template.
        
        Returns:
            True if the model's tokenizer has a chat template.
        """
        try:
            tokenizer = self.llm.get_tokenizer()
            return getattr(tokenizer, "chat_template", None) is not None
        except Exception:
            return False
    
    @staticmethod
    def _format_messages_as_prompt(messages: list[dict[str, str]]) -> str:
        """Format chat messages into a plain-text prompt for non-chat models.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            
        Returns:
            Formatted prompt string.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role}: {content}")
        # Add generation prompt
        parts.append("Assistant:")
        return "\n".join(parts)
    
    async def generate(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Generate completion for chat messages.
        
        Args:
            request: CompletionRequest with messages, model, temperature, etc.
            
        Returns:
            OpenAI-compatible completion response with format:
            {
                "id": "cmpl-...",
                "object": "chat.completion",
                "created": <timestamp>,
                "model": <model_name>,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": <text>},
                    "finish_reason": "stop"
                }]
            }
            
        Raises:
            RuntimeError: If generation fails
        """

        logger.info(f"Generating completion for model: {request.model}")
        logger.debug(f"Request: {request}")
        
        try:
            # Convert messages to OpenAI format for vLLM
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Create sampling parameters from request
            # Use defaults for optional parameters
            temperature = request.temperature if request.temperature is not None else 0.7
            max_tokens = request.max_tokens if request.max_tokens is not None else 512
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,  # Default nucleus sampling
            )
            
            logger.debug(f"Sampling params: {sampling_params}")
            
            # Use chat interface if supported, otherwise fall back to generate
            if self._supports_chat:
                outputs = self.llm.chat(
                    messages=messages,  # type: ignore[arg-type]
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            else:
                prompt = self._format_messages_as_prompt(messages)
                logger.debug(f"Using plain generate with prompt: {prompt[:100]}...")
                outputs = self.llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            
            # Extract generated text from output
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"Successfully generated {len(generated_text)} characters")
            
            # Format response in OpenAI-compatible format
            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            
            return response
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def health_check(self) -> bool:
        """
        Perform health check by attempting a simple generation.
        
        Returns:
            True if engine is healthy, False otherwise
        """

        try:
            logger.debug("Performing health check")
            
            # Try a minimal generation
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=5
            )
            
            # Use a simple prompt for health check
            if self._supports_chat:
                messages = [{"role": "user", "content": "Hello"}]
                outputs = self.llm.chat(
                    messages=messages,  # type: ignore[arg-type]
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            else:
                outputs = self.llm.generate(
                    prompts=["Hello"],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            
            # Check if we got a valid response
            if outputs and len(outputs) > 0 and outputs[0].outputs:
                logger.info("Health check passed")
                return True
            else:
                logger.warning("Health check failed: no output generated")
                return False
                
        except Exception as e:
            logger.error(f"Health check failed with exception: {str(e)}", exc_info=True)
            return False
