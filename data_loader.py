# data_loader.py

import pandas as pd
import os


def load_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def load_all_data(processed_dir):
    data_files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
    data = {}
    for file in data_files:
        key = file.replace("synthetic_", "").replace("_data.csv", "")
        data[key] = load_csv(os.path.join(processed_dir, file))
    return data


if __name__ == "__main__":
    processed_dir = "./data/processed"
    all_data = load_all_data(processed_dir)
    for key, df in all_data.items():
        print(f"Loaded {key} with {len(df)} records.")
