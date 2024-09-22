# tests/test_synthetic_data_generator.py

import pytest
import pandas as pd
from data_generation.synthetic_data_generator import generate_synthetic_customer_data


def test_generate_synthetic_customer_data():
    df = generate_synthetic_customer_data(100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "customer_id" in df.columns
    assert "age" in df.columns
    assert "income" in df.columns
    assert "spending_score" in df.columns
    # Add more assertions as needed
