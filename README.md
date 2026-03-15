# Customer Churn Intelligence System

An end-to-end, production-style **Customer Churn Prediction and Retention Intelligence** project for portfolio/resume use.

## 1) Project Overview
This repository simulates a real industry ML system rather than a notebook-only experiment. It includes:
- Data ingestion and synthetic telecom churn dataset generation
- Data cleaning and feature engineering
- Multi-model training and comparison (Logistic Regression, Random Forest, XGBoost)
- Evaluation with business-relevant metrics
- Explainability with SHAP (best-effort execution)
- Retention strategy recommendation engine
- Tableau-ready analytics dataset
- FastAPI prediction service
- Reproducible pipeline scripts

## 2) Repository Structure
```text
customer-churn-intelligence
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   └── exploratory_analysis.ipynb
│
├── src
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── pipeline.py
│   └── utils.py
│
├── models
│   └── churn_model.pkl            # generated
│
├── dashboard
│   └── tableau_dataset.csv        # generated
│
├── api
│   └── app.py
│
├── requirements.txt
└── README.md
```

## 3) Dataset
The project generates a realistic Telco-style churn dataset with core telecom dimensions:
- Demographics: gender, senior citizen, partner, dependents
- Account data: tenure, contract, payment method, paperless billing, monthly/total charges
- Service usage: internet service, streaming, tech support, device protection, etc.
- Target: `Churn` (Yes/No)

Raw and cleaned datasets are saved to:
- `data/raw/telco_churn.csv`
- `data/processed/telco_churn_clean.csv`

## 4) Feature Engineering
Additional intelligence features are created:
- **CLV proxy** = `tenure * MonthlyCharges`
- **Service engagement score** = number of services used
- **Risk indicator** = high monthly charge + short tenure
- **Contract stability score** = numerical encoding of contract lock-in
- **Tenure bucket** for dashboard/cohort analysis

## 5) Model Training
Training workflow:
1. Stratified train/test split
2. Preprocessing with imputation, scaling, and one-hot encoding
3. Candidate models:
   - Logistic Regression (class weighting)
   - Random Forest (class weighting)
   - XGBoost (primary model)
4. Hyperparameter tuning via `GridSearchCV` and cross-validation
5. Best model selected by ROC-AUC and persisted in `models/churn_model.pkl`

## 6) Model Evaluation
Metrics generated:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

Why accuracy alone is not enough:
- Churn datasets are usually imbalanced.
- A high-accuracy model can still miss many churners (false negatives), which is costly for retention teams.
- Precision/Recall/F1 and ROC-AUC better measure the trade-off for intervention decisions.

Outputs:
- `models/evaluation/metrics.json`
- `models/evaluation/confusion_matrix.png`
- `models/evaluation/roc_curve.png`
- `models/evaluation/shap_summary.png` (if SHAP runs successfully)

## 7) Explainability
SHAP-based explanations identify key churn drivers such as:
- Short tenure
- Month-to-month contracts
- Higher monthly charges
- Lack of tech support/device protection

## 8) Retention Intelligence Layer
The recommendation engine transforms churn risk into actions:
- `prob > 0.7` + month-to-month → discount + annual migration offer
- `prob > 0.7` + tenure < 6 → onboarding intervention
- `0.5 < prob <= 0.7` → loyalty bundle intervention
- otherwise → engagement/upsell strategy

## 9) Prediction Pipeline
`src/predict.py` provides callable inference:
- Input: customer feature payload
- Output:
  - churn probability
  - risk level (Low/Medium/High)
  - recommended retention action

## 10) API Service
FastAPI service in `api/app.py`:
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

## 11) Tableau Dashboard Dataset
`dashboard/tableau_dataset.csv` includes:
- churn probability
- tenure / tenure bucket
- monthly charges
- contract type
- predicted churn
- revenue_at_risk

Dashboard design ideas:
- Churn rate by contract type
- Churn by tenure bucket
- High-risk segment map
- Revenue at risk by segment
- Retention opportunity funnel

## 12) How to Run
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run full pipeline
```bash
python -m src.pipeline
```

### Run prediction script
```bash
python -m src.predict
```

### Launch API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Example API request
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

## 13) Advanced Enhancements (Roadmap)
- Scheduled retraining pipeline (cron/Airflow/GitHub Actions)
- MLflow experiment tracking and model registry
- Segment discovery via clustering
- Cohort-based churn survival tracking
- Feature drift and data quality monitoring
