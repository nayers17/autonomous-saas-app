# pipelines/seo_keyword_optimization_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.seo_keyword_optimization_model import SEOKeywordOptimizationModel


def run_seo_keyword_optimization_pipeline():
    """
    Executes the SEO Keyword Optimization pipeline.
    """
    logger = setup_logger("seo_keyword_optimization.log")
    logger.info("SEO Keyword Optimization pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_seo_keyword_optimization_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = SEOKeywordOptimizationModel()
        logger.info("Initialized SEOKeywordOptimizationModel.")

        # Generate keywords for each input
        generated_keywords = []
        for idx, row in data.iterrows():
            document = row["keyword"]
            keywords = model.generate_keywords(document)
            generated_keywords.append(keywords)
            if idx % 50 == 0:
                logger.info(f"Generated keywords for input {idx}.")

        # Save generated keywords (Placeholder)
        generated_data = pd.DataFrame({"generated_keywords": generated_keywords})
        generated_data_path = "data/processed/generated_keywords.csv"
        generated_data.to_csv(generated_data_path, index=False)
        logger.info(f"Saved generated keywords to {generated_data_path}.")

        logger.info("SEO Keyword Optimization pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in SEO Keyword Optimization pipeline: {e}",
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_seo_keyword_optimization_pipeline()
