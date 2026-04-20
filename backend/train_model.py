from __future__ import annotations

# ============================================================================
# FRAUD DETECTION TRAINING PIPELINE
#
# Main techniques:
# - preprocessing
# - feature engineering
# - EDA
# - classification models
# - imbalance handling
# - evaluation metrics
# - feature importance
# - ETL + star schema + OLAP analysis
# ============================================================================

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_integration import build_etl_tables, build_olap_views
from feature_engineering import FraudFeatureEngineer
from preprocessing import (
    ARTIFACTS_DIR,
    STANDARD_DATASET_PATH,
    TARGET_COLUMN,
    identify_column_types,
    load_standardized_dataset,
    preprocess_transactions,
    split_features_target,
)

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None

BASE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DIR = ARTIFACTS_DIR / "warehouse"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model_bundle.pkl"
REPORT_PATH = ARTIFACTS_DIR / "fraud_lifecycle_report.json"
TRAINING_SAMPLE_PATH = ARTIFACTS_DIR / "training_sample.csv"


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_native(v) for v in value]
    if isinstance(value, tuple):
        return [to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return 0.0 if int(y_true.sum()) == 0 else float(average_precision_score(y_true, y_prob))


def metrics_from_probs(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
    }


def build_eda_payload(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    class_distribution = df[TARGET_COLUMN].value_counts().sort_index()
    amount_edges = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000]
    amount_bands = pd.cut(df["Amount"], bins=amount_edges, include_lowest=True, right=False)
    amount_counts = amount_bands.value_counts(sort=False)
    amount_labels = [
        f"₹{int(amount_edges[i] // 1000)}k-₹{int(amount_edges[i + 1] // 1000)}k"
        for i in range(len(amount_edges) - 1)
    ]

    by_type = df.groupby(["TransactionType", TARGET_COLUMN]).size().unstack(fill_value=0).reset_index()
    by_location = df.groupby(["Location", TARGET_COLUMN]).size().unstack(fill_value=0).reset_index()

    trend = (
        df.assign(month=df["TransactionDate"].dt.to_period("M").astype(str))
        .groupby("month")[TARGET_COLUMN]
        .sum()
        .reset_index()
        .sort_values("month")
    )

    fraud_by_merchant = df.groupby("MerchantID")[TARGET_COLUMN].sum().sort_values(ascending=False).head(10)
    numeric_corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)[TARGET_COLUMN].dropna()

    return {
        "class_distribution": {
            "labels": ["Not Fraud (0)", "Fraud (1)"],
            "values": [int(class_distribution.get(0, 0)), int(class_distribution.get(1, 0))],
        },
        "amount_distribution": {
            "labels": amount_labels,
            "values": amount_counts.astype(int).tolist(),
        },
        "fraud_vs_nonfraud_by_transaction_type": {
            "labels": by_type["TransactionType"].astype(str).tolist(),
            "non_fraud": by_type.get(0, pd.Series([0] * len(by_type))).astype(int).tolist(),
            "fraud": by_type.get(1, pd.Series([0] * len(by_type))).astype(int).tolist(),
        },
        "fraud_vs_nonfraud_by_location": {
            "labels": by_location["Location"].astype(str).tolist(),
            "non_fraud": by_location.get(0, pd.Series([0] * len(by_location))).astype(int).tolist(),
            "fraud": by_location.get(1, pd.Series([0] * len(by_location))).astype(int).tolist(),
        },
        "fraud_trend_over_time": {
            "labels": trend["month"].astype(str).tolist(),
            "values": trend[TARGET_COLUMN].astype(int).tolist(),
        },
        "fraud_by_merchant": {
            "labels": fraud_by_merchant.index.astype(str).tolist(),
            "values": fraud_by_merchant.astype(int).tolist(),
        },
        "numeric_feature_correlation_with_target": {
            "labels": numeric_corr.index.astype(str).tolist(),
            "values": [round(float(v), 4) for v in numeric_corr.values],
        },
    }


def train_test_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COLUMN],
        random_state=42,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df[TARGET_COLUMN],
        random_state=42,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def threshold_sweep(y_true: pd.Series, y_prob: np.ndarray, min_thr: float = 0.01, max_thr: float = 0.20) -> Tuple[float, List[Dict[str, Any]]]:
    thresholds = np.arange(min_thr, max_thr + 0.0001, 0.01)
    rows: List[Dict[str, Any]] = []
    best_threshold = float(min_thr)
    best_score = -1.0

    for threshold in thresholds:
        metrics = metrics_from_probs(y_true, y_prob, float(threshold))
        rows.append(metrics)
        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, rows


def build_curves(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, Any]:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    if len(np.unique(y_true)) < 2:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_thresholds = np.array([np.inf, 0.0])
    else:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    return {
        "pr_curve": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "thresholds": [float(v) for v in pr_thresholds],
        },
        "roc_curve": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) for v in roc_thresholds],
        },
    }


def top_k_capture(y_true: pd.Series, y_prob: np.ndarray, k_percent: float) -> Dict[str, float]:
    k = max(1, int(len(y_true) * (k_percent / 100.0)))
    ranked = np.argsort(y_prob)[::-1]
    top_idx = ranked[:k]
    y_arr = y_true.to_numpy()
    fraud_total = int(y_arr.sum())
    fraud_captured = int(y_arr[top_idx].sum())
    capture_rate = 0.0 if fraud_total == 0 else fraud_captured / fraud_total
    precision_at_k = fraud_captured / len(top_idx)
    return {
        "k_percent": float(k_percent),
        "capture_rate": float(capture_rate),
        "precision_at_k": float(precision_at_k),
        "fraud_captured": fraud_captured,
        "fraud_total": fraud_total,
    }


def build_pca_payload(X_matrix: Any, y: pd.Series, sample_size: int = 3000) -> Dict[str, Any]:
    if len(y) == 0:
        return {"explained_variance_ratio": [], "fraud": [], "non_fraud": []}

    if len(y) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X_matrix,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=42,
        )
    else:
        X_sample = X_matrix
        y_sample = y

    dense_sample = X_sample.toarray() if hasattr(X_sample, "toarray") else np.asarray(X_sample)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(dense_sample)
    fraud_points: List[Dict[str, float]] = []
    non_fraud_points: List[Dict[str, float]] = []
    y_array = y_sample.to_numpy()

    for idx, point in enumerate(components):
        payload_point = {"x": float(point[0]), "y": float(point[1])}
        if int(y_array[idx]) == 1:
            fraud_points.append(payload_point)
        else:
            non_fraud_points.append(payload_point)

    return {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "fraud": fraud_points,
        "non_fraud": non_fraud_points,
    }


def train() -> Dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    standardized_df = load_standardized_dataset()
    cleaned_result = preprocess_transactions(standardized_df)
    clean_df = cleaned_result.df
    clean_df["TransactionDate"] = pd.to_datetime(clean_df["TransactionDate"], errors="coerce")

    data_understanding = {
        "column_types": identify_column_types(clean_df),
        "target_variable": TARGET_COLUMN,
        "target_distribution": clean_df[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict(),
        "class_imbalance_ratio": round(
            float(clean_df[TARGET_COLUMN].value_counts().max() / clean_df[TARGET_COLUMN].value_counts().min()),
            4,
        ),
        "inspection_summary": {
            "amount_min": float(clean_df["Amount"].min()),
            "amount_max": float(clean_df["Amount"].max()),
            "date_min": str(clean_df["TransactionDate"].min()),
            "date_max": str(clean_df["TransactionDate"].max()),
            "merchant_unique_count": int(clean_df["MerchantID"].nunique()),
            "transaction_type_unique_count": int(clean_df["TransactionType"].nunique()),
            "location_unique_count": int(clean_df["Location"].nunique()),
        },
    }

    etl_tables = build_etl_tables(clean_df)
    for table_name, table_df in etl_tables.items():
        table_df.to_csv(WAREHOUSE_DIR / f"{table_name}.csv", index=False)
    olap_payload = build_olap_views(etl_tables["Fact_Transactions"])

    acquisition_payload = {
        "source_file": str(STANDARD_DATASET_PATH),
        "rows": int(len(clean_df)),
        "columns": clean_df.columns.tolist(),
        "indianized_locations": sorted(clean_df["Location"].dropna().astype(str).unique().tolist()),
        "rupee_scale": "Amounts are represented on an INR scale and displayed in rupees in the UI.",
        "transaction_types": sorted(clean_df["TransactionType"].dropna().astype(str).unique().tolist()),
    }

    model_df = clean_df.copy()
    if len(model_df) > 8_000:
        model_df, _ = train_test_split(
            model_df,
            train_size=8_000,
            stratify=model_df[TARGET_COLUMN],
            random_state=42,
        )
        model_df = model_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    TRAINING_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(TRAINING_SAMPLE_PATH, index=False)

    train_df, val_df, test_df = train_test_val_split(model_df)
    y_train = train_df[TARGET_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    engineer = FraudFeatureEngineer()
    X_train = train_df.drop(columns=[TARGET_COLUMN, "TransactionID"])
    X_val = val_df.drop(columns=[TARGET_COLUMN, "TransactionID"])
    X_test = test_df.drop(columns=[TARGET_COLUMN, "TransactionID"])

    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_val_fe = engineer.transform(X_val)
    X_test_fe = engineer.transform(X_test)

    categorical_features = ["TransactionType", "Location"]
    numeric_features = [column for column in X_train_fe.columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    X_train_matrix = preprocessor.fit_transform(X_train_fe)
    X_val_matrix = preprocessor.transform(X_val_fe)
    X_test_matrix = preprocessor.transform(X_test_fe)
    feature_names = preprocessor.get_feature_names_out().tolist()
    pca_payload = build_pca_payload(X_train_matrix, y_train)

    pos_weight = float((len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1))
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    model_metrics: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}
    validation_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        model.fit(X_train_matrix, y_train)
        fitted_models[model_name] = model
        validation_probs[model_name] = model.predict_proba(X_val_matrix)[:, 1]
        test_probs[model_name] = model.predict_proba(X_test_matrix)[:, 1]
        model_metrics[model_name] = metrics_from_probs(y_test, test_probs[model_name], threshold=0.10)

    # Optional SMOTE comparison on training data only.
    imbalance_comparison: Dict[str, Any] = {"notes": []}
    if SMOTE is not None:
        dense_train = X_train_matrix.toarray() if hasattr(X_train_matrix, "toarray") else X_train_matrix
        dense_val = X_val_matrix.toarray() if hasattr(X_val_matrix, "toarray") else X_val_matrix
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(dense_train, y_train)
        smote_model = LogisticRegression(max_iter=1200, random_state=42)
        smote_model.fit(X_smote, y_smote)
        smote_validation_probs = smote_model.predict_proba(dense_val)[:, 1]
        imbalance_comparison["smote_logistic"] = metrics_from_probs(y_val, smote_validation_probs, threshold=0.10)
    else:
        imbalance_comparison["notes"].append("SMOTE unavailable in environment")

    # Use validation probabilities for threshold tuning, then report test metrics at tuned threshold.
    best_model_name = max(model_metrics.keys(), key=lambda name: model_metrics[name]["pr_auc"])
    best_validation_probs = validation_probs[best_model_name]
    tuned_threshold, threshold_grid = threshold_sweep(y_val, best_validation_probs)
    best_test_probs = test_probs[best_model_name]
    tuned_test_metrics = metrics_from_probs(y_test, best_test_probs, tuned_threshold)
    curves = build_curves(y_test, best_test_probs)
    capture = {
        "top_1_percent": top_k_capture(y_test, best_test_probs, 1.0),
        "top_5_percent": top_k_capture(y_test, best_test_probs, 5.0),
        "top_10_percent": top_k_capture(y_test, best_test_probs, 10.0),
    }

    feature_importance: Dict[str, Any] = {"tree_based": [], "logistic_coefficients": []}
    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        scores = best_model.feature_importances_
        top_idx = np.argsort(scores)[::-1][:20]
        feature_importance["tree_based"] = [
            {"feature": feature_names[i], "importance": float(scores[i])} for i in top_idx
        ]
    logistic_model = fitted_models.get("LogisticRegression")
    if logistic_model is not None and hasattr(logistic_model, "coef_"):
        coefficients = logistic_model.coef_[0]
        top_idx = np.argsort(np.abs(coefficients))[::-1][:20]
        feature_importance["logistic_coefficients"] = [
            {"feature": feature_names[i], "coefficient": float(coefficients[i])} for i in top_idx
        ]

    key_drivers = [
        item["feature"]
        for item in (
            feature_importance["tree_based"]
            if feature_importance["tree_based"]
            else feature_importance["logistic_coefficients"]
        )[:10]
    ]

    warehouse_summary = {
        "fact_table": "Fact_Transactions",
        "dimensions": ["Dim_Date", "Dim_Merchant", "Dim_Location", "Dim_TransactionType"],
        "table_row_counts": {table_name: int(len(table_df)) for table_name, table_df in etl_tables.items()},
    }

    lifecycle_report = {
        "dmw_lifecycle": {
            "processing": {
                "source_acquisition": acquisition_payload,
                "data_understanding": data_understanding,
                "data_preprocessing": cleaned_result.report,
                "feature_engineering": {
                    "features_created": [
                        "hour",
                        "day",
                        "month",
                        "day_of_week",
                        "is_weekend",
                        "amount_log",
                        "normalized_amount",
                        "high_value_flag",
                        "merchant_transaction_count",
                        "merchant_transaction_frequency",
                        "merchant_fraud_rate",
                        "merchant_avg_amount",
                        "merchant_amount_deviation",
                        "location_transaction_count",
                        "location_transaction_frequency",
                        "location_fraud_rate",
                        "location_avg_amount",
                        "location_amount_deviation",
                        "transaction_type_fraud_rate",
                    ],
                    "leakage_control": "Merchant and location aggregates are learned on training data only and applied to validation/test data.",
                },
                "split_strategy": {
                    "type": "stratified_holdout",
                    "train_rows": int(len(train_df)),
                    "validation_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                },
            },
            "pattern_discovery": {
                "eda": build_eda_payload(clean_df),
                "key_fraud_drivers": key_drivers,
            },
            "classification": {
                "models_trained": list(models.keys()),
                "model_metrics_default_threshold": model_metrics,
                "best_model": best_model_name,
                "threshold_tuning": tuned_test_metrics,
                "threshold_grid": threshold_grid,
                "curves": curves,
                "fraud_capture": capture,
                "pca": pca_payload,
                "imbalance_handling": {
                    "class_weight_used": True,
                    "stratified_split": True,
                    "optional_smote_comparison": imbalance_comparison,
                },
                "feature_importance": feature_importance,
            },
            "warehouse": {
                "etl": "Extract CSV -> transform transaction fields -> load processed dataset",
                "star_schema": warehouse_summary,
                "schema_relationships": [
                    "Fact_Transactions.DateKey -> Dim_Date.DateKey",
                    "Fact_Transactions.MerchantKey -> Dim_Merchant.MerchantKey",
                    "Fact_Transactions.LocationKey -> Dim_Location.LocationKey",
                    "Fact_Transactions.TransactionTypeKey -> Dim_TransactionType.TransactionTypeKey",
                ],
                "olap_analysis": olap_payload,
            },
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(to_native(lifecycle_report), report_file, indent=2)

    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_engineer": engineer,
        "threshold": tuned_threshold,
        "feature_names": feature_names,
        "model_kind": "matrix_model",
    }
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    return lifecycle_report


if __name__ == "__main__":
    report = train()
    print(f"Fraud model trained and saved. Best model: {report['dmw_lifecycle']['classification']['best_model']}")
