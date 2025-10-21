import pytest
from src.data_preprocessing import preprocess_data


def test_preprocess_data():
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Churn': [0, 1, 0],
        'Age': [25, 35, 45],
        'TotalSpend': [500, 1500, 2000]
    })

    X, y = preprocess_data(df)

    assert X.shape == (3, 3)
    assert y.shape == (3,)
