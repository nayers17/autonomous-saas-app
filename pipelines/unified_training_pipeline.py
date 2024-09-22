# scripts/unified_training_pipeline.py

import subprocess
import logging
import os

# Set up logging
logging.basicConfig(
    filename="unified_training_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("unified_training_pipeline")


def run_script(script_path):
    """
    Runs a Python script and logs the output.

    Args:
        script_path (str): Path to the Python script.
    """
    logger.info(f"Running script: {script_path}")
    try:
        result = subprocess.run(
            ["python", script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"Script {script_path} output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Script {script_path} warnings/errors:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path} failed with error:\n{e.stderr}")


def main():
    logger.info("Starting Unified Training Pipeline.")

    # Define the order of scripts to run
    scripts = [
        "pipelines/quantize_models.py",
        "pipelines/layer_freezing.py",
        "pipelines/gradient_accumulation.py",
        "pipelines/mixed_precision_training.py",
        "pipelines/data_parallelism.py",
        "pipelines/incremental_training.py",
        "pipelines/hyperparameter_tuning.py",
        "pipelines/faiss_indexing.py",
        "pipelines/rag_pipeline.py",
    ]

    for script in scripts:
        script_full_path = os.path.join(os.getcwd(), script)
        if os.path.exists(script_full_path):
            run_script(script_full_path)
        else:
            logger.error(f"Script {script_full_path} does not exist.")

    logger.info("Unified Training Pipeline completed successfully.")


if __name__ == "__main__":
    main()
