import logging
import os
from h2o.automl import H2OAutoML
import h2o
import pandas as pd
from sklearn.datasets import make_classification

# Initialize H2O
print("Initializing H2O...")
h2o.init()

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "../data/logs/")
os.makedirs(log_dir, exist_ok=True)  # Ensure the logs directory exists
log_file = os.path.join(log_dir, "ab_testing.log")  # Log file for this pipeline

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# Function to generate synthetic data if dataset is too small
def generate_synthetic_data(num_rows=200, num_features=5):
    print(
        f"Generating synthetic data with {num_rows} rows and {num_features} features..."
    )
    logging.info(
        f"Generating synthetic data with {num_rows} rows and {num_features} features."
    )

    # Using make_classification from sklearn to generate synthetic binary classification data
    X, y = make_classification(
        n_samples=num_rows, n_features=num_features, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(num_features)])
    df["target_column"] = y
    logging.info("Synthetic data generated successfully.")
    print("Synthetic data generated.")
    return df


# Perform A/B Testing
def ab_testing(data_path, min_required_rows=200):
    try:
        print("Starting A/B testing pipeline...")
        logging.info("Starting A/B testing pipeline...")

        # Load data for A/B testing
        if os.path.exists(data_path):
            print(f"Data file found at {data_path}, loading...")
            data = h2o.import_file(data_path)
            logging.info("Loaded existing dataset.")
            print("Data loaded successfully.")
        else:
            print(f"Data file not found at {data_path}, generating synthetic data.")
            logging.warning(
                f"Data file not found at {data_path}, generating synthetic data."
            )
            df_synthetic = generate_synthetic_data(
                num_rows=min_required_rows
            )  # Create synthetic data
            data = h2o.H2OFrame(df_synthetic)
            print("Synthetic data loaded into H2OFrame.")

        # Check if data is large enough
        if data.nrows < min_required_rows:
            print(f"Data has only {data.nrows} rows, generating more synthetic data.")
            logging.warning(
                f"Data has only {data.nrows} rows. Generating synthetic data."
            )
            df_synthetic = generate_synthetic_data(num_rows=min_required_rows)
            data = h2o.H2OFrame(df_synthetic)
            print("New synthetic data generated and loaded.")

        data["target_column"] = data[
            "target_column"
        ].asfactor()  # Convert to categorical
        train, test = data.split_frame(ratios=[0.8])

        logging.info("Data loaded successfully for A/B testing.")
        print("Data split into training and testing sets.")

        # Run AutoML for A/B Testing
        print("Starting H2O AutoML...")
        logging.info("Starting H2O AutoML...")

        # Reduce max_runtime_secs for testing
        aml = H2OAutoML(max_runtime_secs=120, nfolds=0)

        print("Running AutoML training...")
        logging.info("Running AutoML training...")

        aml.train(y="target_column", training_frame=train)

        print("AutoML training complete.")
        logging.info("AutoML training complete.")

        # Get the leader model and predict on the test set
        lb = aml.leaderboard
        logging.info("A/B testing AutoML leaderboard created.")
        print("Leaderboard created.")

        predictions = aml.predict(test)

        logging.info("Predictions generated successfully.")
        print("Predictions generated.")
        return predictions
    except Exception as e:
        logging.error(f"Error during A/B testing: {e}")
        print(f"Error during A/B testing: {e}")
        raise e


# Example usage
if __name__ == "__main__":
    # Path to your dataset
    data_path = "../data/input/ab_testing_data.csv"  # Adjust the path if needed

    # Perform A/B Testing
    result = ab_testing(data_path)
    print("A/B Testing complete.")
    print(result)
