import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / \
    "model" / "churn_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained churn prediction model.
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert raw JSON/dict input into a pandas DataFrame
    that matches the training features.
    """
    df = pd.DataFrame([data])
    # Optionally handle missing or derived features
    if "TotalSpendPerTenure" not in df.columns and "TotalSpend" in df and "Tenure" in df:
        df["TotalSpendPerTenure"] = df["TotalSpend"] / (df["Tenure"] + 1)
    return df


def predict_churn(model, input_data: pd.DataFrame):
    """
    Predict churn for the given customer data.
    Returns prediction and probability.
    """
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Churn probability
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }


# Example usage (for quick testing)
if __name__ == "__main__":
    model = load_model()
    sample_input = {
        "Age": 35,
        "Tenure": 3,
        "TotalSpend": 1200,
        "AverageTransactionValue": 400,
        "PurchaseFrequency": 3
    }
    input_df = preprocess_input(sample_input)
    result = predict_churn(model, input_df)
    print(result)
