# test_summarization_translation_model.py

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch


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

    # Create a summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
        clean_up_tokenization_spaces=True,  # Explicitly set to avoid FutureWarning
    )

    # Test input
    text = (
        "Hugging Face is creating a tool that democratizes AI. "
        "The company provides a wide range of models and tools for natural language processing, computer vision, and more. "
        "Their mission is to make AI accessible to everyone, enabling developers and researchers to build innovative applications."
    )

    # Generate summary
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        print("Summarization Result:")
        print(summary[0]["summary_text"])
    except Exception as e:
        print(f"Error during summarization: {e}")

    # Create a translation pipeline (e.g., English to French)
    translator = pipeline(
        "translation_en_to_fr", model=model, tokenizer=tokenizer, device=device
    )

    # Generate translation
    try:
        translation = translator("Hello, how are you?", max_length=40, do_sample=False)
        print("Translation Result:")
        print(translation[0]["translation_text"])
    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == "__main__":
    main()
