# data_validation.py

import pandas as pd
import os


def validate_dataframe(df, task):
    if task in ["sentiment_analysis", "social_media_sentiment_analysis"]:
        assert (
            "text" in df.columns and "label" in df.columns
        ), "Missing required columns."
        assert df["label"].dtype in [int, "int64"], "Label column must be integers."
    elif task == "code_generation":
        assert (
            "prompt" in df.columns and "response" in df.columns
        ), "Missing required columns."
    # Add more validations as needed
    print(f"Validation passed for {task}.")


def validate_all(cleaned_data):
    for task, df in cleaned_data.items():
        validate_dataframe(df, task)


if __name__ == "__main__":
    from data_preprocessing import preprocess_all
    from data_loader import load_all_data

    processed_dir = "./data/processed"
    all_data = load_all_data(processed_dir)
    cleaned_data = preprocess_all(all_data)
    validate_all(cleaned_data)
