import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

data = {
    "CustomerID": [f"C{i:04d}" for i in range(1, N+1)],
    "Age": np.random.randint(18, 70, N),
    "Gender": np.random.choice(["Male", "Female"], N),
    "Tenure": np.random.randint(1, 10, N),
    "AverageTransactionValue": np.random.uniform(100, 600, N).round(2),
    "PurchaseFrequency": np.random.randint(1, 12, N),
    "LastPurchaseDaysAgo": np.random.randint(1, 180, N),
    "TotalVisits": np.random.randint(5, 100, N),
    "MembershipType": np.random.choice(["Basic", "Silver", "Gold", "Platinum"], N, p=[0.5, 0.3, 0.15, 0.05]),
    "Region": np.random.choice(["North", "South", "East", "West"], N),
    "EmailEngagement": np.random.choice(["Low", "Medium", "High"], N, p=[0.3, 0.5, 0.2]),
    "DiscountUsage": np.random.randint(0, 10, N),
}

# Define churn pattern (simple heuristic)
df = pd.DataFrame(data)
df["Churn"] = np.where(
    (df["Tenure"] < 3) & (df["EmailEngagement"] == "Low") & (
        df["LastPurchaseDaysAgo"] > 60),
    1, 0
)

df.to_csv("data/raw/retail_customers.csv", index=False)
print("✅ Generated sample data: data/raw/retail_customers.csv")
