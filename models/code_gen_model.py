# models/code_gen_model.py

from transformers import pipeline


class CodeGenerationModel:
    def __init__(self, device: int = -1):
        # Load a small GPT-2 model for code generation
        self.pipeline = pipeline(
            "text-generation",
            model="distilgpt2",
            device=device,  # Use GPU if available
            # Removed clean_up_tokenization_spaces=True
        )

    def generate_code(self, prompt: str, max_length: int = 50):
        result = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return {"code": result[0]["generated_text"].strip()}
