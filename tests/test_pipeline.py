import pandas as pd
import numpy as np
import os
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def test_merged_dataset_exists():
    """Ensure the preprocessing pipeline outputs the merged dataset."""
    filepath = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    assert os.path.exists(filepath), "Merged dataset not found. Have you run preprocessing.py?"

def test_feature_dataset_exists():
    """Ensure the feature engineering pipeline outputs the expected datasets."""
    featured = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    encoded = os.path.join(PROCESSED_DIR, "encoded_dataset.csv")
    assert os.path.exists(featured), "Featured dataset not found."
    assert os.path.exists(encoded), "Encoded dataset not found."

def test_target_variable_validity():
    """Ensure the target variable does not contain negative delays or nulls."""
    filepath = os.path.join(PROCESSED_DIR, "encoded_dataset.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        assert "delay_minutes" in df.columns, "delay_minutes target is missing."
        assert df["delay_minutes"].isnull().sum() == 0, "Target variable contains nulls."
        assert (df["delay_minutes"] >= 0).all(), "Target variable contains negative delays."

def test_time_features_exist():
    """Verify that temporal features were generated correctly."""
    filepath = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        expected_cols = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]
        for col in expected_cols:
            assert col in df.columns, f"Temporal feature {col} missing."
