# pipelines/data_analysis_pipeline.py

import h2o
from catboost import CatBoostClassifier, Pool
import pandas as pd
import logging

# Initialize logging
logging.basicConfig(
    filename="../data/logs/data_analysis.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# Initialize H2O cluster
def initialize_h2o():
    try:
        h2o.init(max_mem_size="8G")
        logging.info("H2O cluster initialized successfully for Data Analysis.")
    except Exception as e:
        logging.error(f"Error initializing H2O cluster: {e}")
        raise e


initialize_h2o()


# Perform data analysis with CatBoost
def analyze_data(data_path, target_column):
    try:
        # Load data
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path} for Data Analysis.")

        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Initialize CatBoost
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            eval_metric="Accuracy",
            verbose=0,
            task_type="GPU",
        )

        # Fit model
        model.fit(X, y)
        logging.info("CatBoost model training completed.")

        # Feature importance
        feature_importances = model.get_feature_importance()
        features = X.columns
        importance_df = pd.DataFrame(
            {"Feature": features, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        logging.info("Feature importance calculated.")

        return model, importance_df
    except Exception as e:
        logging.error(f"Error during data analysis: {e}")
        raise e


# Example usage
if __name__ == "__main__":
    # Sample data path and target
    sample_data_path = "../data/processed/data_analysis_data.csv"
    target = "target"

    # Create sample data if not exists
    import os

    if not os.path.exists(sample_data_path):
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "target": [0, 1, 0, 1, 0],
            }
        )
        df.to_csv(sample_data_path, index=False)

    model, importance = analyze_data(sample_data_path, target)
    print("Feature Importances:\n", importance)
