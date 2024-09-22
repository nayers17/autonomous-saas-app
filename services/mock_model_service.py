# services/mock_model_service.py

from services.base_model_service import BaseModelService


class MockModelService(BaseModelService):
    def predict_sentiment(self, text: str):
        positive_keywords = ["happy", "love", "excellent", "good", "great"]
        negative_keywords = ["sad", "hate", "poor", "bad", "terrible"]
        score = 0.0
        label = "NEUTRAL"

        for word in positive_keywords:
            if word in text.lower():
                score = 0.99
                label = "POSITIVE"
                break

        for word in negative_keywords:
            if word in text.lower():
                score = 0.99
                label = "NEGATIVE"
                break

        return {"label": label, "score": score}

    def generate_code(self, prompt: str):
        return {"generated_code": "def example_function():\n    return 'Hello, World!'"}

    def generate_lead(self, prompt: str):
        return {
            "generated_lead": "Introducing our new SaaS platform that revolutionizes remote team management by providing real-time collaboration tools and performance analytics."
        }
