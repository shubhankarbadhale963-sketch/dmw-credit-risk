from __future__ import annotations

# ============================================================================
# FRAUD DETECTION API
# - Health, metrics, EDA, evaluation, warehouse, and live prediction endpoints
# - Keeps UI layout unchanged; only response text/content is fraud oriented
# ============================================================================

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import ARTIFACTS_DIR, DATE_COLUMN, RAW_DATA_PATH, STANDARD_DATASET_PATH, load_standardized_dataset

app = FastAPI(title="Fraud Detection API")


def _cors_config() -> tuple[list[str], bool]:
    """Resolve CORS origins from environment for local + deployed frontend.

    ALLOWED_ORIGINS examples:
    - *
    - https://your-app.vercel.app
    - https://your-app.vercel.app,https://preview.vercel.app
    """
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"], False
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return (origins or ["*"]), True


allowed_origins, allow_credentials = _cors_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = ARTIFACTS_DIR / "fraud_model_bundle.pkl"
REPORT_PATH = ARTIFACTS_DIR / "fraud_lifecycle_report.json"


def _load_model_bundle() -> Dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def _load_report() -> Dict[str, Any] | None:
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH, "r", encoding="utf-8") as report_file:
        return json.load(report_file)


def _load_processed_data() -> pd.DataFrame:
    return load_standardized_dataset()


def _base_metrics() -> Dict[str, Any]:
    df = _load_processed_data()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    return {
        "total_transactions": int(len(df)),
        "fraud_rate_percent": round(float(df["IsFraud"].mean() * 100), 3),
        "avg_amount": round(float(df["Amount"].mean()), 2),
        "date_range": {
            "start": str(df[DATE_COLUMN].min()),
            "end": str(df[DATE_COLUMN].max()),
        },
        "unique_merchants": int(df["MerchantID"].nunique()),
        "unique_locations": int(df["Location"].nunique()),
        "transaction_types": sorted(df["TransactionType"].dropna().astype(str).unique().tolist()),
    }


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "status": "API running",
        "dataset": "credit_card_fraud_dataset.csv",
        "standardized_dataset": str(STANDARD_DATASET_PATH),
        "model_loaded": _load_model_bundle() is not None,
        "report_available": _load_report() is not None,
    }


@app.get("/dashboard/metrics")
def dashboard_metrics() -> Dict[str, Any]:
    return _base_metrics()


@app.get("/lifecycle/summary")
def lifecycle_summary() -> Dict[str, Any]:
    report = _load_report()
    if report is None:
        raise HTTPException(status_code=404, detail="Run training first to generate the lifecycle report")
    return report["dmw_lifecycle"]


@app.get("/eda")
def eda_payload() -> Dict[str, Any]:
    report = _load_report()
    if report is None:
        raise HTTPException(status_code=404, detail="EDA report not available. Run training first.")
    return report["dmw_lifecycle"]["pattern_discovery"]["eda"]


@app.get("/model/evaluation")
def model_evaluation() -> Dict[str, Any]:
    report = _load_report()
    if report is None:
        raise HTTPException(status_code=404, detail="Model evaluation not available. Run training first.")
    classification = report["dmw_lifecycle"]["classification"]
    return {
        "models": classification["models_trained"],
        "metrics": classification["model_metrics_default_threshold"],
        "best_model": classification["best_model"],
        "threshold_tuning": classification["threshold_tuning"],
        "threshold_grid": classification.get("threshold_grid", []),
        "curves": classification.get("curves", {}),
        "fraud_capture": classification.get("fraud_capture", {}),
        "pca": classification.get("pca", {}),
        "feature_importance": classification.get("feature_importance", {}),
        "imbalance_handling": classification["imbalance_handling"],
        "key_fraud_drivers": report["dmw_lifecycle"]["pattern_discovery"]["key_fraud_drivers"],
    }


@app.get("/warehouse")
def warehouse_payload() -> Dict[str, Any]:
    report = _load_report()
    if report is None:
        raise HTTPException(status_code=404, detail="Warehouse artifacts not available. Run training first.")
    return report["dmw_lifecycle"]["warehouse"]


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_bundle = _load_model_bundle()
    if model_bundle is None:
        raise HTTPException(status_code=404, detail="Model artifact missing. Run training first.")

    input_df = pd.DataFrame(
        [
            {
                "TransactionDate": payload.get("TransactionDate", "2024-01-01 00:00:00"),
                "Amount": float(payload.get("Amount", 0.0)),
                "MerchantID": str(payload.get("MerchantID", "M0001")),
                "TransactionType": str(payload.get("TransactionType", "purchase")),
                "Location": str(payload.get("Location", "Unknown")).upper(),
                "TransactionID": str(payload.get("TransactionID", "TXN_PREDICT_1")),
            }
        ]
    )

    engineered = model_bundle["feature_engineer"].transform(input_df)
    transformed = model_bundle["preprocessor"].transform(engineered)
    fraud_probability = float(model_bundle["model"].predict_proba(transformed)[:, 1][0])

    if fraud_probability >= 0.70:
        risk_level = "High"
    elif fraud_probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "fraud_probability": round(fraud_probability, 6),
        "fraud_label": "Fraud" if fraud_probability >= float(model_bundle["threshold"]) else "Not Fraud",
        "risk_level": risk_level,
        "threshold": round(float(model_bundle["threshold"]), 4),
        "model": model_bundle["model_name"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}