import os
import subprocess

# Define all the pipelines you want to test
pipelines = [
    "pipelines/ab_testing_pipeline.py",
    # Add other pipelines as necessary
]


# Function to run a pipeline
def run_pipeline(pipeline):
    print(f"Running {pipeline}...")
    try:
        result = subprocess.run(
            ["python", pipeline], check=True, capture_output=True, text=True
        )
        print(f"{pipeline} completed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {pipeline}: {e}")
        print("Output:", e.stdout)
        print("Error Output:", e.stderr)


# Run all pipelines
for pipeline in pipelines:
    run_pipeline(pipeline)
