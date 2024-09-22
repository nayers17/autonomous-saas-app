# pipelines/summarization_translation_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.summarization_translation_model import SummarizationTranslationModel


def run_summarization_translation_pipeline():
    """
    Executes the Summarization & Translation pipeline.
    """
    logger = setup_logger("summarization_translation.log")
    logger.info("Summarization & Translation pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_summarization_translation_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = SummarizationTranslationModel()
        logger.info("Initialized SummarizationTranslationModel.")

        # Summarize and translate each text
        summaries = []
        translations = []
        for idx, row in data.iterrows():
            text = row["text"]
            summary = model.summarize(text)
            translation = model.translate(text, target_language="es")
            summaries.append(summary)
            translations.append(translation)
            if idx % 50 == 0:
                logger.info(f"Processed text {idx}.")

        # Save summaries and translations (Placeholder)
        data["summary"] = summaries
        data["translation"] = translations
        processed_data_path = "data/processed/summarization_translation_results.csv"
        data.to_csv(processed_data_path, index=False)
        logger.info(
            f"Saved summarization and translation results to {processed_data_path}."
        )

        logger.info("Summarization & Translation pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Summarization & Translation pipeline: {e}",
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_summarization_translation_pipeline()
