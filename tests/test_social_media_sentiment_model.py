# test_social_media_sentiment_model.py

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


def main():
    # Ensure the Hugging Face token is set (if accessing a private model)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    model_name = "distilroberta-base"  # Replace with your specific model

    try:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, use_auth_token=hf_token
        )
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Set device to GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # Create a sentiment analysis pipeline with explicit clean_up_tokenization_spaces
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        clean_up_tokenization_spaces=True,  # Explicitly set to avoid FutureWarning
    )

    # Test input
    test_input = "I am extremely happy with the new features in the app!"

    # Get prediction
    try:
        result = sentiment_pipeline(test_input)
        print("Social Media Sentiment Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")


if __name__ == "__main__":
    main()
