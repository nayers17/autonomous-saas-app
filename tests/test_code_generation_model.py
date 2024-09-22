# test_code_generation_model.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pytest


@pytest.mark.skip(reason="Skipping large model test for now.")
def test_large_model_loading():
    # Large model logic here
    pass


def main():
    # Ensure the Hugging Face token is set
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_HUB_TOKEN is not set. Please set it in your environment variables."
        )

    model_name = (
        "bigcode/starcoder"  # Replace with your specific StarCoder model if different
    )

    try:
        # Initialize tokenizer and model with authentication
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=hf_token
        )
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Set device to GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # Create a text generation pipeline
    code_generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )

    # Test input
    prompt = "def fibonacci(n):"

    # Generate code
    try:
        result = code_generator(prompt, max_length=50, num_return_sequences=1)
        print("Generated Code:")
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"Error during code generation: {e}")


if __name__ == "__main__":
    main()
