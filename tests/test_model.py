import pytest
import pandas as pd
import os
import joblib
from src.model import train_model, evaluate_model, save_model, load_model


@pytest.fixture
def training_data():
    X = pd.DataFrame({
        "Age": [25, 35, 45, 55],
        "Tenure": [1, 3, 5, 7],
        "TotalSpend": [500, 1500, 2500, 3500]
    })
    y = [0, 1, 0, 1]
    return X, y


def test_model_training_returns_model(training_data):
    """Ensure the model is trained successfully."""
    X, y = training_data
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")


def test_model_saving_and_loading(tmp_path, training_data):
    """Ensure the model can be saved and reloaded properly."""
    X, y = training_data
    model = train_model(X, y)
    file_path = tmp_path / "test_model.pkl"
    save_model(model, file_path)
    assert os.path.exists(file_path)

    loaded_model = load_model(file_path)
    assert hasattr(loaded_model, "predict")


def test_model_evaluation_outputs_metrics(training_data):
    """Ensure evaluation runs without errors."""
    X, y = training_data
    model = train_model(X, y)
    try:
        evaluate_model(model, X, y)
    except Exception as e:
        pytest.fail(f"Model evaluation raised an exception: {e}")
