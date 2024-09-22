# pipelines/basic_pipeline.py

import sys
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import yaml

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="basic_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("basic_pipeline")


def setup_basic_pipeline():
    """
    Sets up a basic LangChain pipeline integrating essential models.
    """
    logger.info("Setting up basic LangChain pipeline.")

    # Example: Content Generation Pipeline
    model_info = {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "path": "./models/content_generation/",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_info["name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""
        Generate content based on the following input:

        {input_text}

        Generated Content:
        """,
    )

    llm_chain = LLMChain(llm=model, prompt=prompt, tokenizer=tokenizer, verbose=True)

    logger.info("Basic LangChain pipeline set up successfully.")
    return llm_chain


def run_basic_pipeline(llm_chain, input_text: str):
    """
    Runs the basic LangChain pipeline with the given input.

    Args:
        llm_chain: LangChain LLMChain object.
        input_text (str): Input text for content generation.

    Returns:
        str: Generated content.
    """
    logger.info(f"Running basic pipeline with input: {input_text}")
    response = llm_chain.run({"input_text": input_text})
    logger.info(f"Generated Content: {response}")
    return response


if __name__ == "__main__":
    llm_chain = setup_basic_pipeline()
    sample_input = "Create a marketing plan for a new e-commerce platform."
    generated_content = run_basic_pipeline(llm_chain, sample_input)
    print(generated_content)
