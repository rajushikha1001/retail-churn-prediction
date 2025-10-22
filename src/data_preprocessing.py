import pandas as pd
from sklearn.model_selection import train_test_split

filepath = "data/raw/retail_customers.csv"
def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Drop irrelevant columns (example: 'ID')
    df = df.drop(columns=['ID'])

    # Convert categorical columns to numerical
    df = pd.get_dummies(df, drop_first=True)

    # Feature and target split
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
