# services/model_service.py

import torch
from models.sentiment_model import SentimentModel
from models.code_gen_model import CodeGenerationModel
from models.lead_gen_model import LeadGenerationModel


class ModelService:
    def __init__(self):
        # Determine device: 0 for GPU if available, else -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.sentiment_model = SentimentModel(device=self.device)
        self.code_gen_model = CodeGenerationModel(device=self.device)
        self.lead_gen_model = LeadGenerationModel(device=self.device)

    def analyze_sentiment(self, text: str):
        return self.sentiment_model.analyze_sentiment(text)

    def generate_code(self, prompt: str):
        return self.code_gen_model.generate_code(prompt)

    def generate_lead(self, prompt: str):
        return self.lead_gen_model.generate_lead(prompt)
