# models/sentiment_model.py

import warnings
from transformers import pipeline


class SentimentModel:
    def __init__(self, device: int = -1):
        # Suppress FutureWarning from transformers
        warnings.filterwarnings(
            "ignore",
            message="`clean_up_tokenization_spaces` was not set.*",
            category=FutureWarning,
        )

        # Load the DistilBERT model for sentiment analysis
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,  # Use GPU if available
        )

    def analyze_sentiment(self, text: str):
        result = self.pipeline(text)[0]
        return {"sentiment": result["label"], "confidence": round(result["score"], 4)}
