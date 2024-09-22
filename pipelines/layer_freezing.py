# pipelines/layer_freezing.py

import sys
import os
import torch
from transformers import AutoModelForSequenceClassification
import logging

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="layer_freezing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("layer_freezing")


def freeze_layers(model, freeze_until_layer: int = 6):
    """
    Freezes layers of the model up to the specified layer.

    Args:
        model: The pre-trained model.
        freeze_until_layer (int): The layer number up to which layers will be frozen.
    """
    logger.info(f"Freezing layers up to layer {freeze_until_layer}.")
    for name, param in model.named_parameters():
        if "layer." in name:
            layer_num = int(name.split(".")[1])
            if layer_num < freeze_until_layer:
                param.requires_grad = False
                logger.debug(f"Froze parameter: {name}")
    logger.info("Layer freezing completed.")


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    from data_loader import load_datasets  # Ensure this function exists

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    freeze_layers(model, freeze_until_layer=6)

    # Save the model with frozen layers
    model.save_pretrained("models/frozen_distilbert")
    tokenizer.save_pretrained("models/frozen_distilbert")
    logger.info("Model with frozen layers saved.")
