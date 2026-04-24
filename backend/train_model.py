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
from sklearn.base import BaseEstimator, ClassifierMixin
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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


def _configure_macos_openmp_path() -> None:
    if os.name != "posix":
        return
    openmp_candidates = [
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/opt/libomp/lib",
    ]
    existing = [path for path in openmp_candidates if os.path.isdir(path)]
    if not existing:
        return

    for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        current = os.environ.get(key, "")
        current_parts = [part for part in current.split(":") if part]
        merged = []
        for path in existing + current_parts:
            if path not in merged:
                merged.append(path)
        os.environ[key] = ":".join(merged)


_configure_macos_openmp_path()

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

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


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(categorical_features: List[str], numeric_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(with_mean=False), numeric_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column not in {TARGET_COLUMN, "TransactionID"}]


def threshold_sweep(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = 0.50,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    rows: List[Dict[str, Any]] = []

    if len(thresholds) == 0:
        metrics = metrics_from_probs(y_true, y_prob, 0.5)
        return 0.5, [metrics], metrics

    for threshold in thresholds:
        rows.append(metrics_from_probs(y_true, y_prob, float(threshold)))

    candidates = [row for row in rows if row["recall"] >= min_recall]
    if candidates:
        best_row = max(candidates, key=lambda row: (row["precision"], row["f1_score"], -row["threshold"]))
    else:
        best_row = max(rows, key=lambda row: (row["f1_score"], row["precision"], row["recall"]))

    return float(best_row["threshold"]), rows, best_row


class CatBoostNativeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        iterations: int = 350,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 0.3,
        class_weights: List[float] | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.class_weights = class_weights
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("CatBoostNativeClassifier expects a pandas DataFrame input")

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            class_weights=self.class_weights,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=self.random_state,
            verbose=self.verbose,
            allow_writing_files=False,
        )
        preferred_categoricals = ["TransactionType", "Location", "MerchantID"]
        categorical_columns = [column for column in preferred_categoricals if column in X.columns]
        if not categorical_columns:
            categorical_columns = [
                column
                for column in X.columns
                if pd.api.types.is_object_dtype(X[column])
                or pd.api.types.is_categorical_dtype(X[column])
                or pd.api.types.is_string_dtype(X[column])
            ]

        fit_frame = X.copy()
        for column in categorical_columns:
            fit_frame[column] = fit_frame[column].astype(str)
        categorical_indices = [fit_frame.columns.get_loc(column) for column in categorical_columns]

        self.model_.fit(fit_frame, y, cat_features=categorical_indices)
        self.classes_ = np.array([0, 1])
        self.categorical_columns_ = categorical_columns
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CatBoostNativeClassifier expects a pandas DataFrame input")
        predict_frame = X.copy()
        for column in self.categorical_columns_:
            if column in predict_frame.columns:
                predict_frame[column] = predict_frame[column].astype(str)
        return self.model_.predict_proba(predict_frame)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CatBoostNativeClassifier expects a pandas DataFrame input")
        predict_frame = X.copy()
        for column in self.categorical_columns_:
            if column in predict_frame.columns:
                predict_frame[column] = predict_frame[column].astype(str)
        return self.model_.predict(predict_frame)


def build_tree_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return preprocessor.get_feature_names_out().tolist()


def compute_feature_importance(best_model: Any, feature_names: List[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"tree_based": [], "logistic_coefficients": []}

    if hasattr(best_model, "feature_importances_"):
        scores = np.asarray(best_model.feature_importances_)
        top_idx = np.argsort(scores)[::-1][:20]
        payload["tree_based"] = [
            {"feature": feature_names[index], "importance": float(scores[index])}
            for index in top_idx
        ]

    if hasattr(best_model, "coef_"):
        coefficients = np.asarray(best_model.coef_[0])
        top_idx = np.argsort(np.abs(coefficients))[::-1][:20]
        payload["logistic_coefficients"] = [
            {"feature": feature_names[index], "coefficient": float(coefficients[index])}
            for index in top_idx
        ]

    if hasattr(best_model, "get_feature_importance"):
        scores = np.asarray(best_model.get_feature_importance())
        top_idx = np.argsort(scores)[::-1][:20]
        payload["tree_based"] = [
            {"feature": feature_names[index], "importance": float(scores[index])}
            for index in top_idx
        ]

    return payload


def best_driver_list(feature_importance: Dict[str, Any]) -> List[str]:
    ranked = feature_importance["tree_based"] if feature_importance["tree_based"] else feature_importance["logistic_coefficients"]
    return [item["feature"] for item in ranked[:10]]


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

    X_train = train_df.drop(columns=[TARGET_COLUMN, "TransactionID"], errors="ignore")
    X_val = val_df.drop(columns=[TARGET_COLUMN, "TransactionID"], errors="ignore")
    X_test = test_df.drop(columns=[TARGET_COLUMN, "TransactionID"], errors="ignore")

    feature_engineer = FraudFeatureEngineer().fit(X_train, y_train)
    X_train_fe = feature_engineer.transform(X_train)
    X_val_fe = feature_engineer.transform(X_val)
    X_test_fe = feature_engineer.transform(X_test)

    categorical_features = [
        column
        for column in ["TransactionType", "Location", "MerchantID"]
        if column in X_train_fe.columns
    ]
    numeric_features = [column for column in X_train_fe.columns if column not in categorical_features]
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    visual_matrix = preprocessor.fit_transform(X_train_fe)
    pca_payload = build_pca_payload(visual_matrix, y_train)

    pos_weight = float((len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model_availability_notes: List[str] = []

    model_specs: List[Tuple[str, Any, Dict[str, List[Any]], int, bool]] = []
    model_specs.append(
        (
            "LogisticRegression",
            ImbPipeline(
                steps=[
                    ("engineer", FraudFeatureEngineer()),
                    ("preprocessor", build_preprocessor(categorical_features, numeric_features)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            },
            8,
            True,
        )
    )
    model_specs.append(
        (
            "RandomForest",
            ImbPipeline(
                steps=[
                    ("engineer", FraudFeatureEngineer()),
                    ("preprocessor", build_preprocessor(categorical_features, numeric_features)),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 6, 10, 14],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
            10,
            True,
        )
    )

    if XGBClassifier is not None:
        model_specs.append(
            (
                "XGBoost",
                ImbPipeline(
                    steps=[
                        ("engineer", FraudFeatureEngineer()),
                        ("preprocessor", build_preprocessor(categorical_features, numeric_features)),
                        (
                            "model",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                random_state=42,
                                n_jobs=-1,
                                scale_pos_weight=pos_weight,
                            ),
                        ),
                    ]
                ),
                {
                    "model__n_estimators": [180, 250, 350],
                    "model__max_depth": [3, 4, 5],
                    "model__learning_rate": [0.03, 0.05, 0.08],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__min_child_weight": [1, 3, 5],
                    "model__reg_alpha": [0.0, 0.1, 0.5],
                    "model__reg_lambda": [1.0, 2.0, 5.0],
                },
                8,
                False,
            )
        )
    else:
        model_availability_notes.append(
            "XGBoost unavailable in this runtime (missing libxgboost/libomp linkage on macOS)."
        )

    if CatBoostClassifier is not None:
        model_specs.append(
            (
                "CatBoost",
                ImbPipeline(
                    steps=[
                        ("engineer", FraudFeatureEngineer()),
                        (
                            "model",
                            CatBoostNativeClassifier(
                                class_weights=[1.0, pos_weight],
                                random_state=42,
                                verbose=False,
                            ),
                        ),
                    ]
                ),
                {
                    "model__iterations": [250, 350, 500],
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.05, 0.08],
                    "model__l2_leaf_reg": [3.0, 5.0, 7.0],
                    "model__bagging_temperature": [0.0, 0.3, 0.7],
                    "model__random_strength": [0.5, 1.0, 2.0],
                },
                8,
                False,
            )
        )
    else:
        model_availability_notes.append("CatBoost is not installed in this environment.")

    tuned_metrics_by_model: Dict[str, Any] = {}
    default_metrics_by_model: Dict[str, Any] = {}
    validation_threshold_rows: Dict[str, List[Dict[str, Any]]] = {}
    model_rankings: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Any] = {}
    validation_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}

    for model_name, pipeline, param_grid, n_iter, use_parallel_search in model_specs:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="precision",
            cv=cv,
            random_state=42,
            n_jobs=-1 if use_parallel_search else 1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        fitted_pipelines[model_name] = best_pipeline
        validation_probs[model_name] = best_pipeline.predict_proba(X_val)[:, 1]
        test_probs[model_name] = best_pipeline.predict_proba(X_test)[:, 1]

        validation_threshold, threshold_rows, validation_best_metrics = threshold_sweep(y_val, validation_probs[model_name])
        validation_threshold_rows[model_name] = threshold_rows
        tuned_test_metrics = metrics_from_probs(y_test, test_probs[model_name], validation_threshold)
        default_test_metrics = metrics_from_probs(y_test, test_probs[model_name], 0.5)

        tuned_metrics_by_model[model_name] = tuned_test_metrics
        default_metrics_by_model[model_name] = default_test_metrics
        model_rankings.append(
            {
                "model": model_name,
                "best_params": to_native(search.best_params_),
                "validation_precision": float(validation_best_metrics["precision"]),
                "validation_recall": float(validation_best_metrics["recall"]),
                "validation_f1": float(validation_best_metrics["f1_score"]),
                "validation_roc_auc": safe_roc_auc(y_val, validation_probs[model_name]),
                "validation_pr_auc": safe_pr_auc(y_val, validation_probs[model_name]),
                "validation_threshold": float(validation_threshold),
                "test_default_metrics": default_test_metrics,
                "test_tuned_metrics": tuned_test_metrics,
            }
        )

    if SMOTE is not None:
        smote_preprocessor = build_preprocessor(categorical_features, numeric_features)
        smote_train_matrix = smote_preprocessor.fit_transform(X_train_fe)
        smote_val_matrix = smote_preprocessor.transform(X_val_fe)
        dense_smote_train = smote_train_matrix.toarray() if hasattr(smote_train_matrix, "toarray") else smote_train_matrix
        dense_smote_val = smote_val_matrix.toarray() if hasattr(smote_val_matrix, "toarray") else smote_val_matrix
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(dense_smote_train, y_train)
        smote_model = LogisticRegression(max_iter=2000, random_state=42)
        smote_model.fit(X_smote, y_smote)
        smote_validation_probs = smote_model.predict_proba(dense_smote_val)[:, 1]
        imbalance_comparison: Dict[str, Any] = {
            "smote_logistic": metrics_from_probs(y_val, smote_validation_probs, threshold=0.5),
            "notes": ["SMOTE comparison is evaluated on the validation split only."] ,
        }
    else:
        imbalance_comparison = {"notes": ["SMOTE unavailable in environment"]}

    best_model_name = max(
        model_rankings,
        key=lambda row: (row["validation_precision"], row["validation_recall"], row["validation_pr_auc"]),
    )["model"]
    best_rank = next(row for row in model_rankings if row["model"] == best_model_name)
    best_threshold = float(best_rank["validation_threshold"])
    best_test_probs = test_probs[best_model_name]
    best_tuned_test_metrics = tuned_metrics_by_model[best_model_name]
    curves = build_curves(y_test, best_test_probs)
    capture = {
        "top_1_percent": top_k_capture(y_test, best_test_probs, 1.0),
        "top_5_percent": top_k_capture(y_test, best_test_probs, 5.0),
        "top_10_percent": top_k_capture(y_test, best_test_probs, 10.0),
    }

    best_pipeline = fitted_pipelines[best_model_name]
    best_model = best_pipeline.named_steps["model"]
    if "preprocessor" in best_pipeline.named_steps:
        best_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        pca_source_matrix = best_pipeline.named_steps["preprocessor"].transform(X_train_fe)
    else:
        best_feature_names = X_train_fe.columns.tolist()
        pca_source_preprocessor = build_preprocessor(categorical_features, numeric_features)
        pca_source_matrix = pca_source_preprocessor.fit_transform(X_train_fe)

    feature_importance = compute_feature_importance(best_model, best_feature_names)
    key_drivers = best_driver_list(feature_importance)

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
                        "is_night_transaction",
                        "hour_sin",
                        "hour_cos",
                        "day_of_week_sin",
                        "day_of_week_cos",
                        "month_sin",
                        "month_cos",
                        "amount_log",
                        "normalized_amount",
                        "amount_zscore",
                        "high_value_flag",
                        "amount_to_global_avg_ratio",
                        "merchant_transaction_count",
                        "merchant_transaction_frequency",
                        "merchant_fraud_rate",
                        "merchant_avg_amount",
                        "merchant_amount_deviation",
                        "amount_to_merchant_avg_ratio",
                        "merchant_risk_amount_interaction",
                        "location_transaction_count",
                        "location_transaction_frequency",
                        "location_fraud_rate",
                        "location_avg_amount",
                        "location_amount_deviation",
                        "amount_to_location_avg_ratio",
                        "location_risk_amount_interaction",
                        "transaction_type_fraud_rate",
                        "transaction_type_amount_interaction",
                    ],
                    "leakage_control": "Merchant, location, and transaction-type fraud-rate features are fit inside the training pipeline and smoothed toward the global training fraud rate.",
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
                "models_trained": [item["model"] for item in model_rankings],
                "model_availability_notes": model_availability_notes,
                "model_metrics_default_threshold": default_metrics_by_model,
                "model_metrics_tuned_threshold": tuned_metrics_by_model,
                "best_model": best_model_name,
                "best_threshold": best_threshold,
                "threshold_tuning": best_tuned_test_metrics,
                "threshold_grid": validation_threshold_rows[best_model_name],
                "curves": curves,
                "fraud_capture": capture,
                "pca": build_pca_payload(pca_source_matrix, y_train),
                "imbalance_handling": {
                    "class_weight_used": True,
                    "stratified_split": True,
                    "optional_smote_comparison": imbalance_comparison,
                },
                "feature_importance": feature_importance,
                "model_rankings": model_rankings,
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

    model_bundle = {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_engineer": best_pipeline.named_steps["engineer"],
        "preprocessor": best_pipeline.named_steps.get("preprocessor"),
        "model": best_model,
        "feature_names": best_feature_names,
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(to_native(lifecycle_report), report_file, indent=2)

    return lifecycle_report


if __name__ == "__main__":
    report = train()
    print(f"Fraud model trained and saved. Best model: {report['dmw_lifecycle']['classification']['best_model']}")
