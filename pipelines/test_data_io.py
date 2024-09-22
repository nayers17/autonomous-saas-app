# pipelines/test_data_io.py

import pandas as pd
from pipelines.customer_segmentation_pipeline import segment_customers

# Sample data input for customer segmentation
sample_data = {
    "customer_data": [
        "Enjoys hiking and outdoor activities.",
        "Interested in healthy eating and wellness.",
    ]
}

df = pd.DataFrame(sample_data)

# Call the segmentation pipeline
result = segment_customers(df, n_clusters=2)

# Print the resulting DataFrame
print(result)
