# pipelines/content_generation_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.content_generation_model import ContentGenerationModel

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def run_content_generation_pipeline():
    """
    Executes the Content Generation pipeline.
    """
    logger = setup_logger("content_generation.log")
    logger.info("Content Generation pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_content_generation_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = ContentGenerationModel()
        logger.info("Initialized ContentGenerationModel.")

        # Generate content for each input
        generated_contents = []
        for idx, row in data.iterrows():
            prompt = row["content"]
            content = model.generate_content(prompt)
            generated_contents.append(content)
            if idx % 10 == 0:
                logger.info(f"Generated content sample {idx}.")

        # Save generated content (Placeholder)
        generated_data = pd.DataFrame({"generated_content": generated_contents})
        generated_data_path = "data/processed/generated_content.csv"
        generated_data.to_csv(generated_data_path, index=False)
        logger.info(f"Saved generated content to {generated_data_path}.")

        logger.info("Content Generation pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Content Generation pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_content_generation_pipeline()
