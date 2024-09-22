# pipelines/customer_segmentation_pipeline.py

import sys
import os

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.customer_segmentation_model import CustomerSegmentationModel
import logging

# Set up logging
logging.basicConfig(
    filename="customer_segmentation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("customer_segmentation")


def run_customer_segmentation():
    logger.info("Customer Segmentation pipeline started.")
    # Initialize and run your model
    model = CustomerSegmentationModel()
    model.train()  # Assuming you have a train method
    logger.info("Customer Segmentation pipeline completed successfully.")


if __name__ == "__main__":
    run_customer_segmentation()
