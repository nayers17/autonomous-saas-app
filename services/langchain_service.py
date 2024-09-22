# services/langchain_service.py

from langchain import LLMChain, PromptTemplate
from transformers import pipeline
import torch


class LangChainService:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1

        # Sentiment Analysis Pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
        )

        # Code Generation Pipeline
        self.code_pipeline = pipeline(
            "text-generation",
            model="distilgpt2",
            device=self.device,
            max_length=50,
            num_return_sequences=1,
        )

        # Lead Generation Pipeline
        self.lead_pipeline = pipeline(
            "text2text-generation",
            model="t5-small",
            device=self.device,
            max_length=50,
            num_return_sequences=1,
        )

        # Define Prompt Templates
        self.sentiment_template = PromptTemplate(
            input_variables=["text"],
            template="Analyze the sentiment of the following text:\n\n{text}",
        )

        self.code_template = PromptTemplate(
            input_variables=["prompt"],
            template="Generate Python code for the following task:\n\n{prompt}",
        )

        self.lead_template = PromptTemplate(
            input_variables=["prompt"],
            template="Generate a marketing lead based on the following information:\n\n{prompt}",
        )

        # Define Chains
        self.sentiment_chain = LLMChain(
            llm=self.sentiment_pipeline,
            prompt=self.sentiment_template,
            output_key="sentiment_result",
        )

        self.code_chain = LLMChain(
            llm=self.code_pipeline, prompt=self.code_template, output_key="code_result"
        )

        self.lead_chain = LLMChain(
            llm=self.lead_pipeline, prompt=self.lead_template, output_key="lead_result"
        )

    def analyze_sentiment(self, text: str):
        result = self.sentiment_chain.run({"text": text})
        return result

    def generate_code(self, prompt: str):
        result = self.code_chain.run({"prompt": prompt})
        return result

    def generate_lead(self, prompt: str):
        result = self.lead_chain.run({"prompt": prompt})
        return result
