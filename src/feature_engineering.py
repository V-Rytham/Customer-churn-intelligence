from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


SERVICE_COLUMNS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["CLV_proxy"] = data["tenure"] * data["MonthlyCharges"]

    binary_services = []
    for col in SERVICE_COLUMNS:
        s = data[col].replace({"No internet service": "No", "No phone service": "No"})
        binary_services.append((s == "Yes").astype(int))
    data["service_engagement_score"] = np.sum(binary_services, axis=0)

    data["high_charge_short_tenure_risk"] = (
        (data["MonthlyCharges"] > data["MonthlyCharges"].median()) & (data["tenure"] < 12)
    ).astype(int)

    data["contract_stability_score"] = data["Contract"].map(
        {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2,
        }
    )

    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    data["tenure_bucket"] = pd.cut(data["tenure"], bins=bins, labels=labels, include_lowest=True)

    return data
