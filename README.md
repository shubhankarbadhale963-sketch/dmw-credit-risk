## DMW Fraud Lifecycle Project

This project now uses the new dataset:

- `data/credit_card_fraud_dataset.csv`

The raw fraud CSV has been Indianized in place:

- Transaction amounts are treated as INR-scale values
- Locations are normalized to Indian city names
- Fraud labels are uplifted to a higher fraud ratio for modeling

Legacy credit-risk model artifacts were removed. The project now trains a fresh fraud-classification model and exposes all required lifecycle outputs through API endpoints used by the frontend.

For lighter uploads, generated folders can be removed from the project tree and regenerated as needed:

- `backend/artifacts/`
- `backend/__pycache__/`
- `backend/catboost_info/`

Run training to regenerate artifacts when needed.

## Run Backend

```bash
cd backend
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Training writes:

- `backend/artifacts/fraud_model_bundle.pkl`
- `backend/artifacts/fraud_lifecycle_report.json`
- `backend/artifacts/warehouse/*.csv`

## Run Frontend

Open `frontend/index.html` in a browser (or serve it with any static file server).

## Required Techniques Coverage

### Data understanding

- Column type identification
- Target variable (`IsFraud`) inspection
- Class imbalance ratio
- Inspection of Amount, TransactionDate, MerchantID, TransactionType, Location

### Data preprocessing

- Convert `TransactionDate` to datetime
- Duplicate check and removal
- Missing value check and handling
- `TransactionID` treated as identifier (not predictive feature)
- Merchant encoding decision documented and applied

### Feature engineering

- Date parts: hour/day/month/day_of_week
- `is_weekend`
- High-value transaction flag
- Merchant transaction count
- Merchant fraud rate
- Location fraud rate
- Transaction-type fraud rate

### EDA

- Class distribution
- Amount distribution
- Fraud vs non-fraud by transaction type
- Fraud vs non-fraud by location
- Fraud trend over time
- Fraud by merchant
- Numeric correlation analysis with target

### Classification

- Logistic Regression
- Random Forest

### Imbalanced data handling

- Class weights
- Stratified train-test split
- Threshold tuning
- Optional SMOTE comparison
- Optional undersampling comparison

### Encoding and transformation

- One-hot encoding for `TransactionType` and `Location`
- Scaling for Logistic Regression (via shared preprocessor)

### Model evaluation

- Confusion matrix
- Precision, Recall, F1-score
- ROC-AUC, PR-AUC

### Feature interpretation

- Tree-based feature importances
- Logistic regression coefficient interpretation
- Key fraud drivers list

### Data warehouse techniques

- ETL pipeline (extract CSV, transform, load)
- Star schema tables:
	- `Fact_Transactions`
	- `Dim_Date`
	- `Dim_Merchant`
	- `Dim_Location`
	- `Dim_TransactionType`
- OLAP-style analysis:
	- fraud by date
	- fraud by merchant
	- fraud by location
	- fraud by transaction type

## API Endpoints

- `GET /` health
- `GET /dashboard/metrics`
- `GET /lifecycle/summary`
- `GET /eda`
- `GET /model/evaluation`
- `GET /warehouse`
- `POST /predict`
