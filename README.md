# Customer Churn Intelligence System

> Predictive machine learning pipeline with XGBoost, SHAP interpretability, automated retention strategy routing, and FastAPI serving.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?style=flat-square)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Model_Explainability-blue?style=flat-square)](https://shap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## Overview

Most churn projects stop at fitting a default classifier in a Jupyter notebook and reporting vanity accuracy. On imbalanced churn datasets (15–25% churn), a naive model predicting zero churn gets 80%+ accuracy while letting all at-risk customers leave.

I built this project to simulate an end-to-end production ML workflow:
- **Cost-Sensitive Evaluation**: Evaluating against ROC-AUC and Recall rather than naive accuracy.
- **Model Explainability**: Integrating SHAP values to identify why a customer is flagged for churn.
- **Actionable Retention Layer**: Mapping predicted probabilities directly into automated retention recommendations.
- **REST Serving**: Packaging the model behind a FastAPI inference endpoint.

---

## Pipeline Architecture

```mermaid
flowchart TD
    RawData["Telco Customer Dataset"] --> Preprocess["Data Cleaning & Imputation"]
    
    subgraph FeatureEng ["Feature Engineering"]
        Preprocess --> CLV["CLV Proxy: tenure × MonthlyCharges"]
        Preprocess --> Engage["Service Engagement Score"]
        Preprocess --> Risk["Price-to-Tenure Risk Ratio"]
        Preprocess --> Stability["Contract Lock-in Index"]
    end

    subgraph Modeling ["Model Training & Tuning"]
        CLV & Engage & Risk & Stability --> Split["Stratified Train / Test Split"]
        Split --> LogReg["Logistic Regression (Class-Weighted)"]
        Split --> RF["Random Forest Classifier"]
        Split --> XGB["XGBoost Classifier (GridSearchCV)"]
    end

    subgraph Evaluation ["Evaluation & Explainability"]
        XGB --> Metrics["Evaluation: ROC-AUC, Recall, Precision, F1"]
        XGB --> SHAPExplainer["SHAP TreeExplainer"]
    end

    subgraph Delivery ["Serving Layer"]
        Metrics --> BestModel[("Serialized Model: churn_model.pkl")]
        BestModel --> RecEngine["Retention Rules Engine"]
        RecEngine --> API["FastAPI Endpoint: /predict"]
        RecEngine --> BIExport["Tableau / Analytics Dataset"]
    end
```

---

## Key Highlights

- **Engineered Domain Features**: Created custom signals including Customer Lifetime Value (CLV) proxy, multi-service adoption score, and contract stability weight.
- **Model Comparison**: Benchmarked Logistic Regression, Random Forest, and tuned XGBoost with stratified cross-validation.
- **SHAP Feature Attribution**: Surfaces primary macro churn drivers (month-to-month contracts, lack of tech support, fiber-optic pricing friction) for business visibility.
- **Retention Strategy Rules**: Maps probability buckets to practical retention interventions (e.g., annual contract migration discounts for short-tenure month-to-month accounts).

---

## Project Structure

```text
customer-churn-intelligence/
├── api/
│   └── app.py                  # FastAPI REST endpoints (/predict, /health)
├── data/
│   ├── raw/                    # Raw Telco CSV data
│   └── processed/              # Preprocessed datasets
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA & feature distributions
├── src/
│   ├── data_processing.py      # Imputation, encoding, train/test split
│   ├── feature_engineering.py  # Domain feature calculation
│   ├── train_model.py          # GridSearchCV tuning and model export
│   ├── evaluate_model.py       # ROC curves, confusion matrices, SHAP
│   ├── predict.py              # Inference pipeline
│   └── pipeline.py             # Reproducible pipeline runner
└── requirements.txt
```

---

## Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/V-Rytham/Customer-churn-intelligence.git
cd Customer-churn-intelligence

python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Training Pipeline
```bash
python -m src.pipeline
```

### 3. Launch Prediction API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs available at `http://localhost:8000/docs`.

### 4. Sample Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 260.0
  }'
```
