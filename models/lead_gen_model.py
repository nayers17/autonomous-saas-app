# models/lead_gen_model.py

from transformers import pipeline


class LeadGenerationModel:
    def __init__(self, device: int = -1):
        # Load a small T5 model for text generation tasks
        self.pipeline = pipeline(
            "text2text-generation",
            model="t5-small",
            device=device,  # Use GPU if available
            # Removed clean_up_tokenization_spaces=True
        )

    def generate_lead(self, prompt: str, max_length: int = 50):
        result = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return {
            "lead": result[0]["generated_text"].strip(),
            "confidence": 0.90,  # Dummy confidence score
        }
