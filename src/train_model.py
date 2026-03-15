from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import load_processed_data, split_data
from src.feature_engineering import add_engineered_features
from src.utils import save_artifact, save_json

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


def get_models():
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {"model__n_estimators": [200], "model__max_depth": [8, 12, None]},
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_estimators=300,
            ),
            {
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        )
    else:
        warnings.warn("xgboost is unavailable. Install xgboost for primary model.")
    return models


def train_and_select_best():
    df = add_engineered_features(load_processed_data())
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = get_models()

    best_name = None
    best_score = -np.inf
    best_estimator = None
    model_results = {}

    for name, (model, param_grid) in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)
        y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
        result = {
            "best_params": grid.best_params_,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        model_results[name] = result

        if result["roc_auc"] > best_score:
            best_score = result["roc_auc"]
            best_name = name
            best_estimator = grid.best_estimator_

    bundle = {
        "model": best_estimator,
        "feature_columns": X_train.columns.tolist(),
        "best_model_name": best_name,
        "model_results": model_results,
    }

    save_artifact(bundle, "models/churn_model.pkl")
    save_json({"best_model": best_name, "results": model_results}, "models/training_summary.json")

    # Create scored dataset for dashboarding and BI.
    scored_df = df.copy()
    scored_df["churn_probability"] = best_estimator.predict_proba(df.drop(columns=["Churn", "customer_id"]))[:, 1]
    scored_df["predicted_churn"] = np.where(scored_df["churn_probability"] > 0.5, "Yes", "No")
    scored_df["revenue_at_risk"] = np.where(scored_df["predicted_churn"] == "Yes", scored_df["MonthlyCharges"], 0)
    Path("dashboard").mkdir(exist_ok=True, parents=True)
    scored_df.to_csv("dashboard/tableau_dataset.csv", index=False)
    return bundle


if __name__ == "__main__":
    trained = train_and_select_best()
    print(f"Training complete. Best model: {trained['best_model_name']}")
