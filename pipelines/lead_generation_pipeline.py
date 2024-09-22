# pipelines/lead_generation_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.lead_generation_model import LeadGenerationModel


def run_lead_generation_pipeline():
    """
    Executes the Lead Generation pipeline.
    """
    logger = setup_logger("lead_generation.log")
    logger.info("Lead Generation pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_lead_generation_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = LeadGenerationModel()
        logger.info("Initialized LeadGenerationModel.")

        # Generate leads for each input
        generated_leads = []
        for idx, row in data.iterrows():
            input_data = row["lead"]
            leads = model.generate_leads(input_data)
            generated_leads.append(leads)
            if idx % 50 == 0:
                logger.info(f"Generated leads for input {idx}.")

        # Save generated leads (Placeholder)
        generated_data = pd.DataFrame({"generated_leads": generated_leads})
        generated_data_path = "data/processed/generated_leads.csv"
        generated_data.to_csv(generated_data_path, index=False)
        logger.info(f"Saved generated leads to {generated_data_path}.")

        logger.info("Lead Generation pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Lead Generation pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_lead_generation_pipeline()
