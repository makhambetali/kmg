import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


INPUT_CSV = "semantic_risk_dataset.csv"
MODEL_DIR = "model_artifacts"
TARGET_COL = "incident_in_next_14d"
TIME_COL = "event_time"
RANDOM_STATE = 42

# Можно поменять потом
CATEGORICAL_COLS = [
    "organization",
    "observation_type",
    "observation_category",
]

NUMERIC_BASE_COLS = [
    "is_risk_signal",
    "is_good_practice",
    "work_stopped_flag",
    "reported_to_responsible_flag",
    "hazard_eliminated_flag",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend",
    "past_korgau_count_1d",
    "past_korgau_count_7d",
    "past_korgau_count_14d",
    "past_risk_signal_count_7d",
    "past_risk_signal_count_14d",
]

DROP_COLS = {
    "korgau_id",
    "signal_text",
    "region",
    "location",
    "observer_name",
    "event_time",
    "future_incident_count_7d",
    "future_severe_incident_count_7d",
    "future_max_severity_7d",
    "future_mean_severity_7d",
    "future_min_days_to_incident_7d",
    "future_mean_days_to_incident_7d",
    "future_incident_count_14d",
    "future_severe_incident_count_14d",
    "future_max_severity_14d",
    "future_mean_severity_14d",
    "future_min_days_to_incident_14d",
    "future_mean_days_to_incident_14d",
    "future_max_similarity_7d",
    "future_mean_similarity_7d",
    "future_top1_similarity_7d",
    "future_top3_mean_similarity_7d",
    "future_similar_incident_count_7d",
    "future_max_similarity_14d",
    "future_mean_similarity_14d",
    "future_top1_similarity_14d",
    "future_top3_mean_similarity_14d",
    "future_similar_incident_count_14d",
    "incident_in_next_7d",
    "incident_in_next_14d",
    "severe_incident_in_next_14d",
    "semantically_similar_incident_in_next_14d",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if TIME_COL not in df.columns:
        raise KeyError(f"В файле нет колонки '{TIME_COL}'")
    if TARGET_COL not in df.columns:
        raise KeyError(f"В файле нет target колонки '{TARGET_COL}'")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def get_embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    emb_cols = get_embedding_cols(df)

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    numeric_cols = [c for c in NUMERIC_BASE_COLS if c in df.columns]
    numeric_cols.extend(emb_cols)

    # Если появились дополнительные числовые колонки, которые не target/leakage - тоже добавим
    for col in df.columns:
        if col.startswith("future_"):
            continue
        if col in DROP_COLS:
            continue
        if col == TARGET_COL:
            continue
        if col in categorical_cols or col in numeric_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return categorical_cols, numeric_cols


def time_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=10,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def pick_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best = {
        "threshold": 0.5,
        "precision": None,
        "recall": None,
        "f1": -1.0,
    }

    # thresholds length = len(precisions)-1
    for i, thr in enumerate(thresholds):
        p = precisions[i]
        r = recalls[i]
        if (p + r) == 0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)

        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }

    return best


def evaluate_split(name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    result = {
        "split": name,
        "size": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": None,
        "pr_auc": None,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        result["pr_auc"] = float(average_precision_score(y_true, y_proba))

    return result


def save_predictions(df: pd.DataFrame, y_true: np.ndarray, y_proba: np.ndarray, threshold: float, path: str) -> None:
    out = df.copy()
    out["target"] = y_true
    out["pred_proba"] = y_proba
    out["pred_label"] = (y_proba >= threshold).astype(int)

    cols = [
        c for c in [
            "korgau_id",
            "event_time",
            "organization",
            "observation_type",
            "observation_category",
            "signal_text",
            "target",
            "pred_proba",
            "pred_label",
        ] if c in out.columns
    ]
    out = out[cols].sort_values("pred_proba", ascending=False)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ensure_dir(MODEL_DIR)

    df = load_data(INPUT_CSV)

    print("Loaded dataset:", df.shape)
    print("Target distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))

    categorical_cols, numeric_cols = build_feature_lists(df)

    print("\nCategorical cols:")
    print(categorical_cols)

    print("\nNumeric cols count:", len(numeric_cols))

    feature_cols = categorical_cols + numeric_cols
    if not feature_cols:
        raise ValueError("Не найдено фичей для обучения.")

    # Убираем строки без target
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    train_df, valid_df, test_df = time_split(df)

    print("\nTime split sizes:")
    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COL].values

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    pipe = build_pipeline(categorical_cols, numeric_cols)

    print("\nTraining model...")
    pipe.fit(X_train, y_train)

    print("Scoring validation...")
    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    best_thr = pick_best_threshold(y_valid, valid_proba)

    threshold = best_thr["threshold"]
    print("\nBest threshold from validation:")
    print(best_thr)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_split("train", y_train, train_proba, threshold)
    valid_metrics = evaluate_split("valid", y_valid, valid_proba, threshold)
    test_metrics = evaluate_split("test", y_test, test_proba, threshold)

    metrics = {
        "target_col": TARGET_COL,
        "threshold_selection": best_thr,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "numeric_columns_count": len(numeric_cols),
        "train_time_range": {
            "min": str(train_df[TIME_COL].min()),
            "max": str(train_df[TIME_COL].max()),
        },
        "valid_time_range": {
            "min": str(valid_df[TIME_COL].min()),
            "max": str(valid_df[TIME_COL].max()),
        },
        "test_time_range": {
            "min": str(test_df[TIME_COL].min()),
            "max": str(test_df[TIME_COL].max()),
        },
    }

    model_path = os.path.join(MODEL_DIR, "risk_model.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    features_path = os.path.join(MODEL_DIR, "feature_columns.json")
    test_preds_path = os.path.join(MODEL_DIR, "test_predictions.csv")
    valid_preds_path = os.path.join(MODEL_DIR, "valid_predictions.csv")

    print("\nSaving artifacts...")
    joblib.dump(pipe, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    save_predictions(valid_df, y_valid, valid_proba, threshold, valid_preds_path)
    save_predictions(test_df, y_test, test_proba, threshold, test_preds_path)

    print("\nDone.")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved features: {features_path}")
    print(f"Saved valid predictions: {valid_preds_path}")
    print(f"Saved test predictions: {test_preds_path}")

    print("\n=== VALID METRICS ===")
    print(json.dumps(valid_metrics, ensure_ascii=False, indent=2))

    print("\n=== TEST METRICS ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()