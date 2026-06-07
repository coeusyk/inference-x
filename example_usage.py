"""Example usage of VLLMEngine for testing Phase 1 implementation."""
import asyncio
import logging
from pathlib import Path

from src.core.engine import VLLMEngine
from src.core.schemas import CompletionRequest, Message
from src.utils.config_loader import get_model_by_name


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Demonstrate VLLMEngine usage."""
    
    # Load model configuration.
    # - opt-125m:      base model only, no instruction following
    # - qwen2.5-0.5b:  instruction-tuned, ungated, good for testing  <-- default
    # - tinyllama-chat: instruction-tuned, ungated, ~1.1B
    # - llama3-8b:     requires `huggingface-cli login`
    print("Loading model configuration...")
    model_config = get_model_by_name("qwen2.5-0.5b")
    print(f"Loaded config for: {model_config['name']}")
    
    # Initialize engine
    print("\nInitializing VLLMEngine...")
    engine = VLLMEngine(model_config)
    
    # Perform health check
    print("\nPerforming health check...")
    is_healthy = engine.health_check()
    print(f"Health check: {'✓ PASSED' if is_healthy else '✗ FAILED'}")
    
    if not is_healthy:
        print("Engine is not healthy, exiting...")
        return
    
    # Create a completion request
    print("\nGenerating completion...")
    request = CompletionRequest(
        messages=[
            Message(role="system", content="You are a helpful AI assistant."),
            Message(role="user", content="What is the capital of France?")
        ],
        model="qwen2.5-0.5b",
        max_tokens=100,
        temperature=0.7
    )
    
    # Generate response
    response = await engine.generate(request)
    
    # Display results
    print("\n" + "="*80)
    print("RESPONSE")
    print("="*80)
    print(f"Model: {response['model']}")
    print(f"ID: {response['id']}")
    print(f"\nGenerated text:")
    print(response['choices'][0]['message']['content'])
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
