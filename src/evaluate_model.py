from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_processing import load_processed_data, split_data
from src.feature_engineering import add_engineered_features
from src.utils import load_artifact, save_json


def evaluate(model_path: str = "models/churn_model.pkl"):
    bundle = load_artifact(model_path)
    model = bundle["model"]

    df = add_engineered_features(load_processed_data())
    X_train, X_test, y_train, y_test = split_data(df)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    Path("models/evaluation").mkdir(parents=True, exist_ok=True)
    save_json(metrics, "models/evaluation/metrics.json")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    fig.tight_layout()
    fig.savefig("models/evaluation/confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("models/evaluation/roc_curve.png")
    plt.close(fig)

    # SHAP explainability (best effort)
    try:
        import shap

        transformed = model.named_steps["preprocessor"].transform(X_test)
        clf = model.named_steps["model"]

        sample_n = min(500, transformed.shape[0])
        idx = np.random.default_rng(42).choice(transformed.shape[0], sample_n, replace=False)
        transformed_sample = transformed[idx]

        explainer = shap.Explainer(clf)
        shap_values = explainer(transformed_sample)

        fig = plt.figure()
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.tight_layout()
        fig.savefig("models/evaluation/shap_summary.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        save_json({"warning": f"SHAP step skipped: {exc}"}, "models/evaluation/shap_warning.json")

    return metrics


if __name__ == "__main__":
    out = evaluate()
    print(out)
