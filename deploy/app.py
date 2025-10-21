from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load pre-trained model
model = joblib.load("model/churn_model.pkl")


class CustomerData(BaseModel):
    age: int
    tenure: int
    total_spend: float
    # Add other features...


@app.post("/predict/")
def predict(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    return {"churn_prediction": prediction[0]}
