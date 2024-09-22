# services/base_model_service.py

from abc import ABC, abstractmethod


class BaseModelService(ABC):
    @abstractmethod
    def predict_sentiment(self, text: str):
        pass

    @abstractmethod
    def generate_code(self, prompt: str):
        pass

    @abstractmethod
    def generate_lead(self, prompt: str):
        pass
