# pipelines/quantize_models.py

import sys
import os
import torch
from transformers import AutoModel
import logging

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="quantization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quantization")


def quantize_model(model_name: str, output_dir: str):
    """
    Quantizes the specified model and saves the quantized version.

    Args:
        model_name (str): Name of the pre-trained model to quantize.
        output_dir (str): Directory to save the quantized model.
    """
    logger.info(f"Loading model '{model_name}' for quantization.")
    try:
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        return

    logger.info("Applying dynamic quantization.")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    try:
        # Save the quantized model state dict
        torch.save(
            quantized_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )
        logger.info(
            f"Quantized model state dict saved to '{output_dir}/pytorch_model.bin'."
        )
    except Exception as e:
        logger.error(f"Error saving quantized model: {e}")


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load model configuration
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Quantize all TensorFlow and PyTorch models that benefit from quantization
    for model_key, model_info in config["models"].items():
        if model_info["framework"] in ["PyTorch", "TensorFlow"]:
            model_to_quantize = model_info["name"]
            quantized_output_path = os.path.join(
                model_info["path"], f"{model_key}-quantized"
            )
            quantize_model(model_to_quantize, quantized_output_path)
