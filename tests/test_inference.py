import pytest
import pandas as pd
from src.predict import load_model, preprocess_input, predict_churn


@pytest.fixture
def mock_input():
    """Sample input for inference."""
    return {
        "Age": 30,
        "Tenure": 2,
        "TotalSpend": 800,
        "AverageTransactionValue": 200,
        "PurchaseFrequency": 4
    }


def test_model_loads_correctly():
    """Ensure model loads without errors."""
    model = load_model()
    assert model is not None, "Model failed to load."


def test_preprocess_input_returns_dataframe(mock_input):
    """Check input preprocessing output."""
    df = preprocess_input(mock_input)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Age" in df.columns


def test_predict_churn_returns_expected_keys(mock_input):
    """Ensure predict_churn output structure is correct."""
    model = load_model()
    df = preprocess_input(mock_input)
    result = predict_churn(model, df)

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert isinstance(result["churn_prediction"], int)
    assert 0.0 <= result["churn_probability"] <= 1.0
