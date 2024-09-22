# pipelines/faiss_indexing.py

import sys
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import yaml

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="faiss_indexing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("faiss_indexing")


def embed_documents(documents, tokenizer, model, device):
    """
    Converts documents to embeddings.

    Args:
        documents (list): List of document texts.
        tokenizer: Tokenizer instance.
        model: Model instance.
        device: Device to perform computation on.

    Returns:
        np.ndarray: Array of embeddings.
    """
    logger.info("Embedding documents.")
    inputs = tokenizer(
        documents, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    logger.info(f"Generated embeddings for {len(documents)} documents.")
    return embeddings


def create_faiss_index(embeddings, index_path: str, use_gpu: bool = False):
    """
    Creates a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        index_path (str): Path to save the FAISS index.
        use_gpu (bool): Whether to use GPU for FAISS indexing.
    """
    dimension = embeddings.shape[1]
    logger.info(f"Creating FAISS index with dimension {dimension}.")

    # Using HNSW index for better performance on large datasets
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors

    if use_gpu and torch.cuda.is_available():
        logger.info("Moving FAISS index to GPU.")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index saved to '{index_path}'.")


if __name__ == "__main__":
    # Example usage
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load documents from a data source
    from data_loader import load_documents  # Ensure this function exists

    documents = load_documents()  # Should return a list of document texts

    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = embed_documents(documents, tokenizer, model, device)
    faiss_index_path = "faiss_index.bin"
    create_faiss_index(
        embeddings, faiss_index_path, use_gpu=False
    )  # Set to True if GPU FAISS is set up
