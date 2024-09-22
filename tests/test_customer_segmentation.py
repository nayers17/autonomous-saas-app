import unittest
import pandas as pd
import pytest
from pipelines.customer_segmentation_pipeline import run_customer_segmentation_pipeline


class TestCustomerSegmentation(unittest.TestCase):

    def test_run_customer_segmentation_pipeline(self):
        # Implement test logic, possibly using mocking
        run_customer_segmentation_pipeline()
        # Add assertions to verify expected outcomes
        # For example, checking if the CSV file was generated or if the clustering was performed correctly
        # assert some condition (e.g., check if output CSV exists, etc.)
        # Example: Assuming the pipeline saves output to a file, you could check if the file exists
        import os

        self.assertTrue(os.path.exists("data/processed/customer_clusters.csv"))


if __name__ == "__main__":
    unittest.main()
