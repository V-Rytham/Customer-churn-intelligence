# Customer Churn Intelligence

A practical, end-to-end machine learning project for telecom churn prediction with a retention recommendation layer.

The goal of this repository is to model what a small production ML project looks like in practice:
- clear folder structure
- modular Python code
- reproducible training/evaluation pipeline
- explainability outputs
- API inference endpoint
- BI-friendly dataset export

---

## What this project does

Given customer profile + account/service information, the system predicts churn probability and returns a retention action recommendation.

Core capabilities:
1. Data ingestion / generation (Telco-style dataset)
2. Cleaning + preprocessing
3. Feature engineering
4. Multi-model training + model selection
5. Evaluation with classification metrics and plots
6. SHAP-based explainability (best effort)
7. Retention intelligence rules
8. Batch-ready scoring export for dashboards
9. FastAPI prediction service

---

## Project layout

```text
customer-churn-intelligence
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── pipeline.py
│   └── utils.py
│
├── models/
├── dashboard/
│   └── tableau_dataset.csv     # generated after training
│
├── api/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses a Telco-like churn schema and generates realistic synthetic records if raw data is not already present.

Target:
- `Churn` (`Yes` / `No`)

Feature groups include:
- demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`)
- account details (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`)
- service usage (`InternetService`, `TechSupport`, `DeviceProtection`, streaming features, etc.)
- billing (`MonthlyCharges`, `TotalCharges`)

Generated files:
- `data/raw/telco_churn.csv`
- `data/processed/telco_churn_clean.csv`

---

## Feature engineering

Implemented engineered features:
- `CLV_proxy = tenure * MonthlyCharges`
- `service_engagement_score`
- `high_charge_short_tenure_risk`
- `contract_stability_score`
- `tenure_bucket`

These are used for both model training and downstream reporting.

---

## Modeling approach

Candidate models:
- Logistic Regression
- Random Forest
- XGBoost (preferred model when available)

Training flow:
1. Stratified train/test split
2. Numeric imputation + scaling
3. Categorical imputation + one-hot encoding
4. Cross-validated hyperparameter search (`GridSearchCV`)
5. Best model selected by ROC-AUC
6. Persist model bundle to `models/churn_model.pkl`

---

## Evaluation

Reported metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Saved artifacts:
- `models/evaluation/metrics.json`
- `models/evaluation/confusion_matrix.png`
- `models/evaluation/roc_curve.png`
- `models/evaluation/shap_summary.png` (if SHAP execution succeeds)

Why not rely on accuracy alone?
- Churn is often class-imbalanced.
- A model can look accurate while still missing many true churners.
- Recall/F1/ROC-AUC better reflect intervention quality for retention use cases.

---

## Retention recommendation engine

The inference layer converts risk into action.

Example business rules:
- high risk + month-to-month contract → retention discount + contract migration offer
- high risk + very low tenure → onboarding/support intervention
- medium risk → loyalty plan / bundled offer
- low risk → standard engagement

---

## API

FastAPI app: `api/app.py`

Endpoints:
- `GET /health`
- `POST /predict`

Example response:

```json
{
  "churn_probability": 0.82,
  "risk_level": "High",
  "recommended_action": "Offer targeted retention discount and migrate to annual contract"
}
```

---

## Dashboard dataset

After training, `dashboard/tableau_dataset.csv` is generated with fields commonly used in churn dashboards, including:
- churn probability
- predicted churn
- tenure and tenure bucket
- contract type
- monthly charges
- revenue at risk

Suggested dashboard cuts:
- churn rate by contract type
- churn by tenure bucket
- high-risk segment distribution
- revenue-at-risk trend and cohorts

---

## How to run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run full pipeline

```bash
python -m src.pipeline
```

### 3) Run a local prediction test

```bash
python -m src.predict
```

### 4) Start API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## Example API call

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

---

## Next improvements

- MLflow experiment tracking + model registry
- scheduled retraining job
- segmentation (clustering) for targeted retention campaigns
- cohort churn monitoring
- drift/data-quality checks in scoring pipeline
