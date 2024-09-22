# services/real_model_service.py

from services.base_model_service import BaseModelService
from transformers import pipeline
import torch


class RealModelService(BaseModelService):
    def __init__(self, device=-1):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="path/to/fine-tuned/distilroberta_sentiment",
            device=device,
        )
        self.code_gen_pipeline = pipeline(
            "text-generation", model="path/to/fine-tuned/codegen", device=device
        )
        self.lead_gen_pipeline = pipeline(
            "text2text-generation",
            model="path/to/fine-tuned/t5_lead_generation",
            device=device,
        )

    def predict_sentiment(self, text: str):
        result = self.sentiment_pipeline(text)
        return result[0]

    def generate_code(self, prompt: str):
        result = self.code_gen_pipeline(prompt, max_length=50, num_return_sequences=1)
        return {"generated_code": result[0]["generated_text"]}

    def generate_lead(self, prompt: str):
        result = self.lead_gen_pipeline(prompt, max_length=100, num_return_sequences=1)
        return {"generated_lead": result[0]["generated_text"]}
