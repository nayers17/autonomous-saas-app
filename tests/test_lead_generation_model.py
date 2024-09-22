# test_lead_generation_model.py

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
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

    model_name = "t5-small"  # Replace with your specific model

    try:
        # Initialize tokenizer and model with authentication
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, use_auth_token=hf_token
        )
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Set device to GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # Create a text generation pipeline
    lead_generator = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, device=device
    )

    # Test input
    prompt = "Generate a lead generation pitch for a new SaaS product that helps businesses manage remote teams."

    # Generate lead
    try:
        result = lead_generator(prompt, max_length=100, num_return_sequences=1)
        print("Generated Lead:")
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"Error during lead generation: {e}")


if __name__ == "__main__":
    main()
