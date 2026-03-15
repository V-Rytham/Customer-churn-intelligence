from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_customer


app = FastAPI(title="Customer Churn Intelligence API", version="1.0.0")


class CustomerPayload(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerPayload):
    result = predict_customer(payload.model_dump())
    return {
        "churn_probability": round(result.churn_probability, 4),
        "risk_level": result.risk_level,
        "recommended_action": result.recommended_action,
    }
