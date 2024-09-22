# data/synthetic_data_generator.py

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def generate_synthetic_ab_testing_data(n_samples=1000):
    """
    Generates synthetic data for A/B Testing.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    data["target"] = y
    return data


def generate_synthetic_ad_optimization_data(n_samples=1000):
    """
    Generates synthetic data for Ad Optimization.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    data["target"] = y
    return data


def generate_synthetic_code_generation_data(n_samples=100):
    """
    Generates synthetic data for Code Generation.
    """
    # For simplicity, we'll use dummy code snippets
    code_snippets = [f"def function_{i}():\n    pass\n" for i in range(n_samples)]
    data = pd.DataFrame({"code": code_snippets})
    return data


def generate_synthetic_content_generation_data(n_samples=100):
    """
    Generates synthetic data for Content Generation.
    """
    content_samples = [f"Sample content {i}" for i in range(n_samples)]
    data = pd.DataFrame({"content": content_samples})
    return data


def generate_synthetic_customer_data(num_customers: int = 1000) -> pd.DataFrame:
    """
    Generates synthetic customer segmentation data.

    Parameters:
    - num_customers (int): Number of synthetic customers to generate.

    Returns:
    - pd.DataFrame: Synthetic customer data.
    """
    np.random.seed(42)  # For reproducibility

    data = {
        "customer_id": range(1, num_customers + 1),
        "age": np.random.randint(18, 70, size=num_customers),
        "income": np.random.randint(30000, 150000, size=num_customers),
        "spending_score": np.random.randint(1, 100, size=num_customers),
        # Add more features as needed
    }

    df = pd.DataFrame(data)
    return df


def generate_synthetic_lead_generation_data(n_samples=500):
    """
    Generates synthetic data for Lead Generation.
    """
    leads = [f"Lead {i}" for i in range(n_samples)]
    data = pd.DataFrame({"lead": leads})
    return data


def generate_synthetic_seo_keyword_optimization_data(n_samples=500):
    """
    Generates synthetic data for SEO Keyword Optimization.
    """
    keywords = [f"keyword_{i}" for i in range(n_samples)]
    data = pd.DataFrame({"keyword": keywords})
    return data


def generate_synthetic_social_media_sentiment_analysis_data(n_samples=500):
    """
    Generates synthetic data for Social Media Sentiment Analysis.
    """
    posts = [f"Social media post {i}" for i in range(n_samples)]
    data = pd.DataFrame({"post": posts})
    return data


def generate_synthetic_summarization_translation_data(n_samples=500):
    """
    Generates synthetic data for Summarization & Translation.
    """
    texts = [f"Text sample {i}" for i in range(n_samples)]
    data = pd.DataFrame({"text": texts})
    return data


if __name__ == "__main__":
    # Generate and save all synthetic data
    ab_data = generate_synthetic_ab_testing_data()
    ab_data.to_csv("data/processed/synthetic_ab_testing_data.csv", index=False)

    ad_data = generate_synthetic_ad_optimization_data()
    ad_data.to_csv("data/processed/synthetic_ad_optimization_data.csv", index=False)

    code_data = generate_synthetic_code_generation_data()
    code_data.to_csv("data/processed/synthetic_code_generation_data.csv", index=False)

    content_data = generate_synthetic_content_generation_data()
    content_data.to_csv(
        "data/processed/synthetic_content_generation_data.csv", index=False
    )

    customer_segmentation_data = generate_synthetic_customer_data()
    customer_segmentation_data.to_csv(
        "data/processed/synthetic_customer_segmentation_data.csv", index=False
    )

    lead_generation_data = generate_synthetic_lead_generation_data()
    lead_generation_data.to_csv(
        "data/processed/synthetic_lead_generation_data.csv", index=False
    )

    seo_keyword_optimization_data = generate_synthetic_seo_keyword_optimization_data()
    seo_keyword_optimization_data.to_csv(
        "data/processed/synthetic_seo_keyword_optimization_data.csv", index=False
    )

    sentiment_analysis_data = generate_synthetic_social_media_sentiment_analysis_data()
    sentiment_analysis_data.to_csv(
        "data/processed/synthetic_social_media_sentiment_analysis_data.csv", index=False
    )

    summarization_translation_data = generate_synthetic_summarization_translation_data()
    summarization_translation_data.to_csv(
        "data/processed/synthetic_summarization_translation_data.csv", index=False
    )

    print("Synthetic data generation completed.")
