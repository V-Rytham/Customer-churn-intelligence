from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.feature_engineering import add_engineered_features
from src.utils import load_artifact


@dataclass
class PredictionOutput:
    churn_probability: float
    risk_level: str
    recommended_action: str


def retention_recommendation(customer: Dict[str, Any], churn_probability: float) -> str:
    contract = customer.get("Contract", "Month-to-month")
    tenure = float(customer.get("tenure", 0))

    if churn_probability > 0.7 and contract == "Month-to-month":
        return "Offer targeted retention discount and migrate to annual contract"
    if churn_probability > 0.7 and tenure < 6:
        return "Launch onboarding intervention with proactive support calls"
    if churn_probability > 0.5:
        return "Enroll in loyalty program with personalized service bundle"
    return "Maintain engagement with upsell/cross-sell campaign"


def risk_level_from_probability(probability: float) -> str:
    if probability > 0.7:
        return "High"
    if probability > 0.4:
        return "Medium"
    return "Low"


def predict_customer(customer: Dict[str, Any], model_path: str = "models/churn_model.pkl") -> PredictionOutput:
    bundle = load_artifact(model_path)
    model = bundle["model"]

    row = pd.DataFrame([customer])
    row = add_engineered_features(row)

    # Align columns with training columns.
    for col in bundle["feature_columns"]:
        if col not in row.columns:
            row[col] = 0
    row = row[bundle["feature_columns"]]

    probability = float(model.predict_proba(row)[0, 1])
    risk_level = risk_level_from_probability(probability)
    recommendation = retention_recommendation(customer, probability)
    return PredictionOutput(probability, risk_level, recommendation)


if __name__ == "__main__":
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 260.0,
    }
    print(predict_customer(sample))
