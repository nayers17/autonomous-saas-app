# services/data_service.py
import pandas as pd
import os


def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def load_all_data(data_dir):
    """Load all CSV files from a directory into a dictionary of DataFrames."""
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    data_dict = {}
    for file in data_files:
        key = file.replace(".csv", "")
        data_dict[key] = load_csv(os.path.join(data_dir, file))
    return data_dict
