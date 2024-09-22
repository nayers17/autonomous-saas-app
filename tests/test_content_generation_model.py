from transformers import AutoTokenizer, AutoModelForCausalLM
import pytest


@pytest.mark.skip(reason="Skipping large model test for now.")
def test_large_model_loading():
    # Large model logic here
    pass


def test_llama_content_generation():
    try:
        # Load the tokenizer and model for LLaMA 3.1 8B Instruct
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Loading model {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Test content generation with a prompt
        prompt = (
            "Explain the importance of artificial intelligence in modern technology."
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        print("Generating content...")

        outputs = model.generate(inputs.input_ids, max_length=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Generated text: {generated_text}")

        # Validation to ensure generation was successful
        assert len(generated_text) > 0, "Model failed to generate content."

        print("Content generation test passed!")

    except Exception as e:
        print(f"Error during content generation test: {e}")


# Run the test when the script is executed
if __name__ == "__main__":
    test_llama_content_generation()
