from __future__ import annotations

# ============================================================================
# FRAUD DATA WAREHOUSE TECHNIQUES
# - ETL to build a star schema from transaction fraud data
# - Fact_Transactions with date, merchant, location, and type dimensions
# - OLAP-style aggregates for dashboard analysis
# ============================================================================

from typing import Dict

import pandas as pd


def build_etl_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    working = df.copy()
    working["TransactionDate"] = pd.to_datetime(working["TransactionDate"], errors="coerce")

    dim_date = (
        working[["TransactionDate"]]
        .drop_duplicates()
        .assign(
            DateKey=lambda x: x["TransactionDate"].dt.strftime("%Y%m%d").astype(str),
            Year=lambda x: x["TransactionDate"].dt.year,
            Month=lambda x: x["TransactionDate"].dt.month,
            Day=lambda x: x["TransactionDate"].dt.day,
            DayOfWeek=lambda x: x["TransactionDate"].dt.dayofweek,
        )
        .dropna()
        .reset_index(drop=True)
    )

    dim_merchant = (
        working[["MerchantID"]]
        .drop_duplicates()
        .assign(MerchantKey=lambda x: x["MerchantID"].astype(str))
        .reset_index(drop=True)
    )

    dim_location = (
        working[["Location"]]
        .drop_duplicates()
        .assign(LocationKey=lambda x: x["Location"].astype(str))
        .reset_index(drop=True)
    )

    dim_transaction_type = (
        working[["TransactionType"]]
        .drop_duplicates()
        .assign(TransactionTypeKey=lambda x: x["TransactionType"].astype(str))
        .reset_index(drop=True)
    )

    fact_transactions = working.assign(
        DateKey=working["TransactionDate"].dt.strftime("%Y%m%d").astype(str),
        MerchantKey=working["MerchantID"].astype(str),
        LocationKey=working["Location"].astype(str),
        TransactionTypeKey=working["TransactionType"].astype(str),
    )[[
        "TransactionID",
        "DateKey",
        "MerchantKey",
        "LocationKey",
        "TransactionTypeKey",
        "Amount",
        "IsFraud",
    ]]

    return {
        "Fact_Transactions": fact_transactions,
        "Dim_Date": dim_date,
        "Dim_Merchant": dim_merchant,
        "Dim_Location": dim_location,
        "Dim_TransactionType": dim_transaction_type,
    }


def build_olap_views(fact_transactions: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    fraud_by_date = (
        fact_transactions.groupby("DateKey")["IsFraud"]
        .mean()
        .sort_index()
        .tail(30)
        .reset_index(name="FraudRate")
    )
    fraud_by_merchant = (
        fact_transactions.groupby("MerchantKey")["IsFraud"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="FraudRate")
    )
    fraud_by_location = (
        fact_transactions.groupby("LocationKey")["IsFraud"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="FraudRate")
    )
    fraud_by_type = (
        fact_transactions.groupby("TransactionTypeKey")["IsFraud"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="FraudRate")
    )

    return {
        "fraud_rate_by_date": {
            "labels": fraud_by_date["DateKey"].astype(str).tolist(),
            "values": [round(float(v), 4) for v in fraud_by_date["FraudRate"].tolist()],
        },
        "fraud_rate_by_merchant": {
            "labels": fraud_by_merchant["MerchantKey"].astype(str).tolist(),
            "values": [round(float(v), 4) for v in fraud_by_merchant["FraudRate"].tolist()],
        },
        "fraud_rate_by_location": {
            "labels": fraud_by_location["LocationKey"].astype(str).tolist(),
            "values": [round(float(v), 4) for v in fraud_by_location["FraudRate"].tolist()],
        },
        "fraud_rate_by_transaction_type": {
            "labels": fraud_by_type["TransactionTypeKey"].astype(str).tolist(),
            "values": [round(float(v), 4) for v in fraud_by_type["FraudRate"].tolist()],
        },
    }
