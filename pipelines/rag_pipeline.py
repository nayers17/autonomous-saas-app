# pipelines/rag_pipeline.py

import sys
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain import PromptTemplate, LLMChain
import logging
import yaml
import torch

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_pipeline")


def setup_retriever(index_path: str):
    """
    Sets up the FAISS retriever.

    Args:
        index_path (str): Path to the FAISS index.

    Returns:
        Retriever object.
    """
    logger.info("Setting up FAISS retriever.")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )
    vector_store = FAISS.load_local(index_path, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    logger.info("FAISS retriever set up successfully.")
    return retriever


def setup_rag_chain(retriever):
    """
    Sets up the RAG pipeline using LangChain.

    Args:
        retriever: FAISS retriever object.

    Returns:
        LangChain LLMChain object.
    """
    logger.info("Setting up RAG chain.")
    model_name = "facebook/rag-sequence-nq"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant. Use the following context to answer the question.
    
        Context:
        {context}
    
        Question:
        {question}
    
        Answer:
        """,
    )

    rag_chain = LLMChain(
        llm=model, prompt=prompt, tokenizer=tokenizer, retriever=retriever, verbose=True
    )
    logger.info("RAG chain set up successfully.")
    return rag_chain


def generate_response(rag_chain, query: str):
    """
    Generates a response using the RAG chain.

    Args:
        rag_chain: LangChain LLMChain object.
        query (str): User query.

    Returns:
        str: Generated response.
    """
    logger.info(f"Generating response for query: {query}")
    response = rag_chain.run({"context": "", "question": query})
    logger.info(f"Generated response: {response}")
    return response


if __name__ == "__main__":
    # Example usage
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    index_path = "faiss_index.bin"
    retriever = setup_retriever(index_path)
    rag_chain = setup_rag_chain(retriever)
    user_query = "How to optimize ad campaigns for e-commerce?"
    response = generate_response(rag_chain, user_query)
    print(response)
