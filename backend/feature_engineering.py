from __future__ import annotations

# ============================================================================
# FRAUD DETECTION FEATURE ENGINEERING
# - Time-based, amount-based, and behavioral transaction features
# - Leakage-safe aggregate encodings learned only from training data
# ============================================================================

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class FraudFeatureEngineer:
    merchant_txn_count_: Dict[str, int] | None = None
    merchant_txn_freq_: Dict[str, float] | None = None
    merchant_fraud_rate_: Dict[str, float] | None = None
    merchant_avg_amount_: Dict[str, float] | None = None
    location_txn_count_: Dict[str, int] | None = None
    location_txn_freq_: Dict[str, float] | None = None
    location_fraud_rate_: Dict[str, float] | None = None
    location_avg_amount_: Dict[str, float] | None = None
    type_fraud_rate_: Dict[str, float] | None = None
    global_fraud_rate_: float = 0.0
    global_avg_amount_: float = 0.0
    high_value_threshold_: float = 0.0

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FraudFeatureEngineer":
        train = df.copy()
        train["__target__"] = y.values

        self.global_fraud_rate_ = float(y.mean())
        self.global_avg_amount_ = float(train["Amount"].mean())
        self.high_value_threshold_ = float(train["Amount"].quantile(0.90))

        self.merchant_txn_count_ = train.groupby("MerchantID").size().astype(int).to_dict()
        self.merchant_txn_freq_ = train["MerchantID"].value_counts(normalize=True).to_dict()
        self.merchant_fraud_rate_ = train.groupby("MerchantID")["__target__"].mean().to_dict()
        self.merchant_avg_amount_ = train.groupby("MerchantID")["Amount"].mean().to_dict()

        self.location_txn_count_ = train.groupby("Location").size().astype(int).to_dict()
        self.location_txn_freq_ = train["Location"].value_counts(normalize=True).to_dict()
        self.location_fraud_rate_ = train.groupby("Location")["__target__"].mean().to_dict()
        self.location_avg_amount_ = train.groupby("Location")["Amount"].mean().to_dict()
        self.type_fraud_rate_ = train.groupby("TransactionType")["__target__"].mean().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.merchant_txn_count_ is None:
            raise ValueError("Feature engineer must be fitted before transform.")

        transformed = df.copy()
        transformed["TransactionDate"] = pd.to_datetime(transformed["TransactionDate"], errors="coerce")

        transformed["hour"] = transformed["TransactionDate"].dt.hour
        transformed["day"] = transformed["TransactionDate"].dt.day
        transformed["month"] = transformed["TransactionDate"].dt.month
        transformed["day_of_week"] = transformed["TransactionDate"].dt.dayofweek
        transformed["is_weekend"] = transformed["day_of_week"].isin([5, 6]).astype(int)

        transformed["amount_log"] = np.log1p(transformed["Amount"])
        transformed["normalized_amount"] = transformed["Amount"] / (transformed["Amount"].median() + 1e-6)
        transformed["high_value_flag"] = (transformed["Amount"] >= self.high_value_threshold_).astype(int)

        transformed["merchant_transaction_count"] = transformed["MerchantID"].map(self.merchant_txn_count_).fillna(1)
        transformed["merchant_transaction_frequency"] = transformed["MerchantID"].map(self.merchant_txn_freq_).fillna(0.0)
        transformed["merchant_fraud_rate"] = transformed["MerchantID"].map(self.merchant_fraud_rate_).fillna(self.global_fraud_rate_)
        transformed["merchant_avg_amount"] = transformed["MerchantID"].map(self.merchant_avg_amount_).fillna(self.global_avg_amount_)
        transformed["merchant_amount_deviation"] = transformed["Amount"] - transformed["merchant_avg_amount"]

        transformed["location_transaction_count"] = transformed["Location"].map(self.location_txn_count_).fillna(1)
        transformed["location_transaction_frequency"] = transformed["Location"].map(self.location_txn_freq_).fillna(0.0)
        transformed["location_fraud_rate"] = transformed["Location"].map(self.location_fraud_rate_).fillna(self.global_fraud_rate_)
        transformed["location_avg_amount"] = transformed["Location"].map(self.location_avg_amount_).fillna(self.global_avg_amount_)
        transformed["location_amount_deviation"] = transformed["Amount"] - transformed["location_avg_amount"]

        transformed["transaction_type_fraud_rate"] = transformed["TransactionType"].map(self.type_fraud_rate_).fillna(self.global_fraud_rate_)

        transformed = transformed.drop(columns=["TransactionDate", "MerchantID"])
        return transformed

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
