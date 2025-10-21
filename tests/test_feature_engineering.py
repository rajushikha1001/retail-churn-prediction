import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import feature_engineering


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "AverageTransactionValue": [200, 400, 150],
        "PurchaseFrequency": [3, 5, 2],
        "Tenure": [1, 4, 0]
    })


def test_feature_engineering_adds_new_features(sample_data):
    """Check that expected new columns are created."""
    engineered = feature_engineering(sample_data)

    expected_cols = {"TotalSpend", "SpendPerTenure", "HighValueCustomer"}
    assert expected_cols.issubset(engineered.columns), \
        f"Missing engineered columns: {expected_cols - set(engineered.columns)}"


def test_total_spend_calculation(sample_data):
    """Ensure TotalSpend is calculated correctly."""
    engineered = feature_engineering(sample_data)
    expected = sample_data["AverageTransactionValue"] * \
        sample_data["PurchaseFrequency"]
    assert np.allclose(engineered["TotalSpend"],
                       expected), "TotalSpend calculation mismatch"


def test_no_nulls_or_infs(sample_data):
    """Ensure no NaN or infinite values are introduced."""
    engineered = feature_engineering(sample_data)
    assert not engineered.isnull().any().any(), "Feature engineering introduced NaNs"
    assert np.isfinite(engineered.select_dtypes(include=[np.number])).all().all(), \
        "Feature engineering introduced infinite values"


def test_idempotency(sample_data):
    """Running feature engineering twice should not change results."""
    once = feature_engineering(sample_data)
    twice = feature_engineering(once)
    pd.testing.assert_frame_equal(once, twice)


def test_high_value_customer_flag(sample_data):
    """Check if high-value customer flag is computed correctly."""
    engineered = feature_engineering(sample_data)
    assert set(engineered["HighValueCustomer"].unique()) <= {0, 1}, \
        "HighValueCustomer should only contain binary values (0 or 1)"
