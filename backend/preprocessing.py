from __future__ import annotations

# ============================================================================
# FRAUD DETECTION PREPROCESSING
# - Standardize raw transaction CSV into a fraud-ready schema
# - Create a synthetic but realistic fraud target for modeling/demo use
# - Clean missing values, duplicates, and type issues
# - Keep identifiers out of modeling
# ============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "credit_card_fraud_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STANDARD_DATASET_PATH = ARTIFACTS_DIR / "fraud_transactions_standardized.csv"

IDENTIFIER_COLUMN = "TransactionID"
TARGET_COLUMN = "IsFraud"
DATE_COLUMN = "TransactionDate"

RAW_USECOLS = [
    "TransactionID",
    "TransactionDate",
    "Amount",
    "MerchantID",
    "TransactionType",
    "Location",
    "IsFraud",
]

INDIAN_LOCATION_MAP = {
    "SAN ANTONIO": "MUMBAI",
    "DALLAS": "DELHI",
    "NEW YORK": "BENGALURU",
    "PHILADELPHIA": "HYDERABAD",
    "LOS ANGELES": "CHENNAI",
    "CHICAGO": "PUNE",
    "HOUSTON": "AHMEDABAD",
    "PHOENIX": "KOLKATA",
    "SAN DIEGO": "SURAT",
    "SAN JOSE": "JAIPUR",
}

TARGET_FRAUD_RATE = 0.09


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    report: Dict[str, object]


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    datetime_cols = [col for col in df.columns if "datetime" in str(df[col].dtype)]
    numeric_cols = [
        col
        for col in df.columns
        if col not in categorical_cols and col not in datetime_cols
    ]
    return {"categorical": categorical_cols, "datetime": datetime_cols, "numeric": numeric_cols}


def _parse_time_column(time_series: pd.Series) -> pd.DataFrame:
    time_str = time_series.fillna(0).astype(int).astype(str).str.zfill(6)
    return pd.DataFrame(
        {
            "hour": time_str.str.slice(0, 2).astype(int),
            "minute": time_str.str.slice(2, 4).astype(int),
            "second": time_str.str.slice(4, 6).astype(int),
        }
    )


def _safe_date_parse(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), errors="coerce")


def _normalize_location(series: pd.Series) -> pd.Series:
    normalized = series.fillna("Unknown").astype(str).str.upper().str.strip()
    return normalized.map(INDIAN_LOCATION_MAP).fillna(normalized)


def _build_transaction_type(amount: pd.Series, balance: pd.Series, hour: pd.Series) -> pd.Series:
    amount = amount.fillna(amount.median())
    hour = hour.fillna(0)
    
    amount_q25 = amount.quantile(0.25)
    amount_q75 = amount.quantile(0.75)
    amount_q90 = amount.quantile(0.90)
    
    conditions = [
        (amount >= amount_q90),
        ((hour <= 5) | (hour >= 22)) & (amount >= amount_q75),
        (amount <= amount_q25),
    ]
    choices = ["transfer", "cash_out", "refund"]
    return pd.Series(np.select(conditions, choices, default="purchase"), index=amount.index)


def _build_fraud_target(


    features: pd.DataFrame,
    base_labels: pd.Series,
    fraud_rate: float = TARGET_FRAUD_RATE,
) -> Tuple[pd.Series, pd.Series]:
    amount = features["Amount"].fillna(features["Amount"].median())
    balance = features["AccountBalance"].fillna(features["AccountBalance"].median())
    ratio = amount / (balance.abs() + 1.0)

    amount_rank = amount.rank(pct=True)
    balance_rank = balance.rank(pct=True)
    high_amount = (amount_rank >= 0.9).astype(float)
    night = features["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(float)
    weekend = features["day_of_week"].isin([5, 6]).astype(float)
    type_risk = features["TransactionType"].isin(["transfer", "cash_out"]).astype(float)
    location_risk = features["Location"].map(
        {
            "MUMBAI": 0.9,
            "DELHI": 0.85,
            "BANGALORE": 0.82,
            "HYDERABAD": 0.78,
            "CHENNAI": 0.76,
        }
    ).fillna(0.55)

    # Small deterministic noise keeps the task realistic while remaining reproducible.
    noise = (
        pd.util.hash_pandas_object(features[["TransactionID"]].astype(str), index=False)
        .astype("uint64")
        .mod(997)
        .astype(float)
        / 997.0
    )

    risk_score = (
        1.9 * ratio
        + 0.9 * amount_rank
        + 0.4 * (1.0 - balance_rank)
        + 0.8 * high_amount
        + 0.8 * night
        + 0.45 * weekend
        + 0.75 * type_risk
        + 0.55 * location_risk
        + 0.18 * noise
    )

    fraud = pd.to_numeric(base_labels, errors="coerce").fillna(0).astype(int).copy()
    current_rate = float(fraud.mean()) if len(fraud) else 0.0
    if current_rate < fraud_rate:
        target_count = max(int(round(len(fraud) * fraud_rate)), int(fraud.sum()) + 1)
        deficit = max(target_count - int(fraud.sum()), 0)
        eligible = fraud[fraud == 0].index
        if len(eligible) > 0 and deficit > 0:
            ranking = risk_score.loc[eligible].sort_values(ascending=False)
            fraud.loc[ranking.head(deficit).index] = 1
    return fraud, risk_score


def standardize_transactions(raw_df: pd.DataFrame, fraud_rate: float = TARGET_FRAUD_RATE) -> PreprocessResult:
    working = raw_df.copy()

    initial_rows = int(len(working))
    missing_before = {k: int(v) for k, v in working.isna().sum().to_dict().items()}
    duplicate_rows = int(working.duplicated().sum())

    working = working.drop_duplicates().reset_index(drop=True)

    date_values = _safe_date_parse(working["TransactionDate"])
    working["TransactionDate"] = (
        date_values
        .fillna(date_values.median())
    )

    working["Amount"] = pd.to_numeric(working["Amount"], errors="coerce")
    working["Location"] = _normalize_location(working["Location"])
    working["MerchantID"] = working["MerchantID"].fillna("M0001").astype(str)
    working["TransactionType"] = _build_transaction_type(
        amount=working["Amount"],
        balance=pd.to_numeric(working.get("Amount", pd.Series(dtype=float)), errors="coerce"),
        hour=working["TransactionDate"].dt.hour.fillna(0).astype(int),
    )

    feature_frame = pd.DataFrame(
        {
            "TransactionID": working["TransactionID"],
            "Amount": working["Amount"],
            "AccountBalance": pd.to_numeric(working["Amount"], errors="coerce"),
            "TransactionType": working["TransactionType"],
            "Location": working["Location"],
            "hour": working["TransactionDate"].dt.hour.fillna(0).astype(int),
            "day_of_week": working["TransactionDate"].dt.dayofweek,
        }
    )
    base_labels = pd.to_numeric(working.get(TARGET_COLUMN, 0), errors="coerce").fillna(0).astype(int)
    working[TARGET_COLUMN], working["FraudScore"] = _build_fraud_target(
        feature_frame,
        base_labels=base_labels,
        fraud_rate=fraud_rate,
    )

    standardized = working[
        [
            "TransactionID",
            "TransactionDate",
            "Amount",
            "MerchantID",
            "TransactionType",
            "Location",
            TARGET_COLUMN,
        ]
    ].copy()

    # Fill any leftover nulls with conservative defaults.
    standardized["TransactionDate"] = pd.to_datetime(standardized["TransactionDate"], errors="coerce")
    standardized["TransactionDate"] = standardized["TransactionDate"].fillna(standardized["TransactionDate"].median())
    standardized["Amount"] = pd.to_numeric(standardized["Amount"], errors="coerce")
    standardized["Amount"] = standardized["Amount"].fillna(standardized["Amount"].median())
    for column in ["MerchantID", "TransactionType", "Location"]:
        standardized[column] = standardized[column].fillna("Unknown").astype(str)
    standardized[TARGET_COLUMN] = pd.to_numeric(standardized[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    missing_after = {k: int(v) for k, v in standardized.isna().sum().to_dict().items()}

    report = {
        "source_path": str(RAW_DATA_PATH),
        "initial_rows": initial_rows,
        "final_rows": int(len(standardized)),
        "duplicate_rows_removed": duplicate_rows,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "target_distribution": standardized[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict(),
        "target_definition": "Source fraud labels with deterministic risk-based uplift when needed for fraud analytics demo",
        "fraud_rate_percent": round(float(standardized[TARGET_COLUMN].mean() * 100), 3),
        "column_types": identify_column_types(standardized),
        "merchant_policy": "MerchantID is a categorical identifier used via aggregates and encoding; not treated as a numeric feature.",
        "location_policy": "Locations are normalized to Indian city names.",
        "amount_policy": "Amounts are kept in INR-scale values from the Indianized raw CSV.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(STANDARD_DATASET_PATH, index=False)

    return PreprocessResult(df=standardized, report=report)


def preprocess_transactions(df: pd.DataFrame) -> PreprocessResult:
    # Generic cleanup for the standardized fraud dataset.
    working = df.copy()
    working = working.drop_duplicates().reset_index(drop=True)

    if DATE_COLUMN in working.columns:
        working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN], errors="coerce")
        if working[DATE_COLUMN].isna().any():
            working[DATE_COLUMN] = working[DATE_COLUMN].fillna(working[DATE_COLUMN].median())
        working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN], errors="coerce")
        if working[DATE_COLUMN].dtype != "datetime64[ns]":
            working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN], errors="coerce")

    missing_before = {k: int(v) for k, v in working.isna().sum().to_dict().items()}
    high_missing = [col for col, value in working.isna().mean().items() if value > 0.5]
    if high_missing:
        working = working.drop(columns=high_missing)

    for column in working.columns:
        if column == TARGET_COLUMN:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)
        elif column == DATE_COLUMN:
            working[column] = pd.to_datetime(working[column], errors="coerce")
        elif pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce")
            working[column] = working[column].fillna(working[column].median())
        else:
            mode_value = working[column].mode(dropna=True)
            fallback = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            working[column] = working[column].fillna(fallback).astype(str)

    missing_after = {k: int(v) for k, v in working.isna().sum().to_dict().items()}
    report = {
        "final_rows": int(len(working)),
        "duplicate_rows_removed": int(len(df) - len(working)),
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
    }
    return PreprocessResult(df=working, report=report)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN].copy()
    drop_cols = [TARGET_COLUMN]
    if IDENTIFIER_COLUMN in df.columns:
        drop_cols.append(IDENTIFIER_COLUMN)
    X = df.drop(columns=drop_cols)
    return X, y


def load_standardized_dataset() -> pd.DataFrame:
    expected_columns = ["TransactionID", "TransactionDate", "Amount", "MerchantID", "TransactionType", "Location", TARGET_COLUMN]
    if STANDARD_DATASET_PATH.exists():
        loaded = pd.read_csv(STANDARD_DATASET_PATH)
        if len(loaded) >= 1000 and all(column in loaded.columns for column in expected_columns):
            loaded[DATE_COLUMN] = pd.to_datetime(loaded[DATE_COLUMN], errors="coerce")
            return loaded
    raw_df = pd.read_csv(RAW_DATA_PATH, usecols=RAW_USECOLS)
    return standardize_transactions(raw_df).df
