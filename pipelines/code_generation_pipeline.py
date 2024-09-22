# pipelines/code_generation_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.code_generation_model import CodeGenerationModel


def run_code_generation_pipeline():
    """
    Executes the Code Generation pipeline.
    """
    logger = setup_logger("code_generation.log")
    logger.info("Code Generation pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_code_generation_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = CodeGenerationModel()
        logger.info("Initialized CodeGenerationModel.")

        # Generate code for each input
        generated_codes = []
        for idx, row in data.iterrows():
            prompt = row["code"]
            code = model.generate_code(prompt)
            generated_codes.append(code)
            if idx % 10 == 0:
                logger.info(f"Generated code snippet {idx}.")

        # Save generated codes (Placeholder)
        generated_data = pd.DataFrame({"generated_code": generated_codes})
        generated_data_path = "data/processed/generated_code.csv"
        generated_data.to_csv(generated_data_path, index=False)
        logger.info(f"Saved generated code to {generated_data_path}.")

        logger.info("Code Generation pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Code Generation pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_code_generation_pipeline()
