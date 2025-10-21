# src/feature_engineering.py
import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features for the churn prediction model.
    """
    df = df.copy()

    # Example derived features
    df["TotalSpend"] = df["AverageTransactionValue"] * df["PurchaseFrequency"]
    df["SpendPerTenure"] = df["TotalSpend"] / (df["Tenure"] + 1)
    df["HighValueCustomer"] = (df["TotalSpend"] > 1000).astype(int)

    return df
