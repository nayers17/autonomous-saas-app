# pipelines/ad_optimization_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.ad_optimization_model import AdOptimizationModel


def run_ad_optimization_pipeline():
    """
    Executes the Ad Optimization pipeline.
    """
    logger = setup_logger("ad_optimization.log")
    logger.info("Ad Optimization pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_ad_optimization_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = AdOptimizationModel()
        logger.info("Initialized AdOptimizationModel.")

        # Train the model
        model.train(data)
        logger.info("Trained AdOptimizationModel.")

        # Predict
        predictions = model.predict(data)
        logger.info(f"Generated {len(predictions)} predictions.")

        # Process results (Placeholder)
        logger.info("Ad Optimization pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Ad Optimization pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_ad_optimization_pipeline()
