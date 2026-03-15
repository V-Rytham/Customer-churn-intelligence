from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir


RANDOM_STATE = 42


def generate_telco_like_dataset(n_samples: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.2])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.5, 0.15])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.35, 0.2, 0.2, 0.25],
    )

    base_charge = rng.normal(65, 20, n_samples).clip(18, 130)
    monthly_charges = (
        base_charge
        + np.where(contract == "Month-to-month", 8, -5)
        + np.where(internet_service == "Fiber optic", 12, 0)
        + np.where(internet_service == "No", -25, 0)
    ).clip(18, 140)

    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    gender = rng.choice(["Male", "Female"], size=n_samples)

    def yn(prob: float) -> np.ndarray:
        return rng.choice(["Yes", "No"], size=n_samples, p=[prob, 1 - prob])

    online_security = yn(0.42)
    online_backup = yn(0.46)
    device_protection = yn(0.49)
    tech_support = yn(0.38)
    streaming_tv = yn(0.52)
    streaming_movies = yn(0.51)

    total_charges = (monthly_charges * tenure + rng.normal(0, 120, n_samples)).clip(0, None)

    score = (
        1.4 * (contract == "Month-to-month").astype(float)
        + 0.8 * (payment_method == "Electronic check").astype(float)
        + 0.9 * (tenure < 12).astype(float)
        + 0.6 * (monthly_charges > 85).astype(float)
        + 0.7 * (tech_support == "No").astype(float)
        + 0.35 * (device_protection == "No").astype(float)
        + 0.2 * (senior_citizen == 1).astype(float)
        - 1.0 * (contract == "Two year").astype(float)
        - 0.5 * (tenure > 36).astype(float)
    )
    prob = 1 / (1 + np.exp(-(score - 1.6)))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_samples)],
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": yn(0.9),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.48, 0.1]),
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges.round(2),
            "TotalCharges": total_charges.round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    missing_mask = rng.choice([True, False], size=n_samples, p=[0.03, 0.97])
    df.loc[missing_mask, "TotalCharges"] = np.nan
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["MonthlyCharges"] * cleaned["tenure"])
    cleaned = cleaned.drop_duplicates(subset=["customer_id"]) 
    return cleaned


def save_raw_and_processed_data(raw_path: str = "data/raw/telco_churn.csv", processed_path: str = "data/processed/telco_churn_clean.csv") -> None:
    ensure_dir(Path(raw_path).parent)
    ensure_dir(Path(processed_path).parent)
    raw_df = generate_telco_like_dataset()
    raw_df.to_csv(raw_path, index=False)
    clean_telco_data(raw_df).to_csv(processed_path, index=False)


def load_processed_data(processed_path: str = "data/processed/telco_churn_clean.csv") -> pd.DataFrame:
    p = Path(processed_path)
    if not p.exists():
        save_raw_and_processed_data()
    return pd.read_csv(p)


def split_data(df: pd.DataFrame, target_col: str = "Churn"):
    X = df.drop(columns=[target_col, "customer_id"])
    y = (df[target_col] == "Yes").astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


if __name__ == "__main__":
    save_raw_and_processed_data()
    print("Saved raw and processed datasets.")
