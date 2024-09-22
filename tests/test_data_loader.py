# tests/test_data_loader.py

import pytest
from data_loader import load_csv


def test_load_csv():
    # Create a sample CSV file
    sample_data = "prompt,response\nprint('Hello'),print('Hello, World!')"
    with open("tests/sample_code.csv", "w") as f:
        f.write(sample_data)

    df = load_csv("tests/sample_code.csv")
    assert len(df) == 1
    assert df.iloc[0]["prompt"] == "print('Hello')"
    assert df.iloc[0]["response"] == "print('Hello, World!')"

    # Clean up
    import os

    os.remove("tests/sample_code.csv")
