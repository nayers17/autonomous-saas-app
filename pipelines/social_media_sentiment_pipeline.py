# pipelines/social_media_sentiment_analysis_pipeline.py

import logging
import os
import sys
import pandas as pd
from utils.logger import setup_logger
from models.social_media_sentiment_analysis_model import (
    SocialMediaSentimentAnalysisModel,
)


def run_social_media_sentiment_analysis_pipeline():
    """
    Executes the Social Media Sentiment Analysis pipeline.
    """
    logger = setup_logger("social_media_sentiment_analysis.log")
    logger.info("Social Media Sentiment Analysis pipeline started.")

    try:
        # Load synthetic data
        data_path = "data/processed/synthetic_social_media_sentiment_analysis_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Synthetic data not found at {data_path}.")
            sys.exit(1)

        data = pd.read_csv(data_path)
        logger.info(f"Loaded synthetic data with shape {data.shape}.")

        # Initialize the model
        model = SocialMediaSentimentAnalysisModel()
        logger.info("Initialized SocialMediaSentimentAnalysisModel.")

        # Analyze sentiment for each post
        sentiments = []
        for idx, row in data.iterrows():
            post = row["post"]
            sentiment = model.analyze_sentiment(post)
            sentiments.append(sentiment)
            if idx % 50 == 0:
                logger.info(f"Analyzed sentiment for post {idx}.")

        # Save sentiments (Placeholder)
        data["sentiment"] = sentiments
        sentiment_data_path = "data/processed/social_media_sentiments.csv"
        data.to_csv(sentiment_data_path, index=False)
        logger.info(f"Saved sentiment analysis results to {sentiment_data_path}.")

        logger.info("Social Media Sentiment Analysis pipeline completed successfully.")

    except Exception as e:
        logger.error(
            f"An error occurred in Social Media Sentiment Analysis pipeline: {e}",
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_social_media_sentiment_analysis_pipeline()
