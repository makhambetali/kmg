import os
import json
import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif


INPUT_CSV = "semantic_risk_dataset.csv"
OUTPUT_DIR = "leakage_debug"
TARGET_COL = "incident_in_next_14d"
TIME_COL = "event_time"
RANDOM_STATE = 42

CATEGORICAL_CANDIDATES = [
    "organization",
    "observation_type",
    "observation_category",
]

SUSPICIOUS_PATTERNS = [
    "future",
    "incident",
    "risk",
    "anomaly",
    "cluster",
    "similar",
    "severity",
    "days_to",
]

SAFE_EXACT_DROP = {
    "korgau_id",
    "event_time",
    "signal_text",
    "region",
    "location",
    "observer_name",
    "incident_in_next_7d",
    "incident_in_next_14d",
    "severe_incident_in_next_14d",
    "semantically_similar_incident_in_next_14d",
}

# Если хочешь ужесточить - можно добавить сюда
ALWAYS_KEEP_NUMERIC_PREFIXES = [
    "emb_",
]

BASE_SAFE_NUMERIC = [
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Нет target колонки: {TARGET_COL}")
    if TIME_COL not in df.columns:
        raise KeyError(f"Нет time колонки: {TIME_COL}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df


def get_embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def is_suspicious(col: str) -> bool:
    lc = col.lower()
    return any(p in lc for p in SUSPICIOUS_PATTERNS)


def classify_columns(df: pd.DataFrame) -> dict:
    all_cols = df.columns.tolist()
    emb_cols = get_embedding_cols(df)

    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in {TARGET_COL}]

    suspicious_cols = [c for c in all_cols if is_suspicious(c)]
    safe_exact = [c for c in all_cols if c in SAFE_EXACT_DROP]

    safe_numeric = [c for c in BASE_SAFE_NUMERIC if c in df.columns]
    safe_numeric += emb_cols

    # Автоматически соберем "другие" числовые
    auto_numeric = []
    for c in numeric_cols:
        if c in safe_numeric:
            continue
        if c in safe_exact:
            continue
        auto_numeric.append(c)

    # Настоящий безопасный набор - исключаем подозрительные
    strict_safe_numeric = []
    for c in numeric_cols:
        if c in SAFE_EXACT_DROP:
            continue
        if is_suspicious(c):
            continue
        strict_safe_numeric.append(c)

    strict_safe_categorical = []
    for c in categorical_cols:
        if c in SAFE_EXACT_DROP:
            continue
        if is_suspicious(c):
            continue
        strict_safe_categorical.append(c)

    return {
        "all_cols": all_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "embedding_cols": emb_cols,
        "suspicious_cols": suspicious_cols,
        "safe_exact_drop": safe_exact,
        "base_safe_numeric": safe_numeric,
        "auto_numeric": auto_numeric,
        "strict_safe_numeric": strict_safe_numeric,
        "strict_safe_categorical": strict_safe_categorical,
    }


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    transformers = []

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if not transformers:
        raise ValueError("Нет колонок для обучения.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=250,
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


def evaluate_model(train_df, valid_df, test_df, categorical_cols, numeric_cols, name: str) -> dict:
    feature_cols = categorical_cols + numeric_cols
    if not feature_cols:
        return {
            "name": name,
            "status": "skipped",
            "reason": "no features",
        }

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COL].values

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    pipe = build_pipeline(categorical_cols, numeric_cols)
    pipe.fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    result = {
        "name": name,
        "status": "ok",
        "n_features": len(feature_cols),
        "categorical_cols": categorical_cols,
        "numeric_cols_count": len(numeric_cols),
        "valid_positive_rate": float(np.mean(y_valid)),
        "test_positive_rate": float(np.mean(y_test)),
        "valid_proba_min": float(np.min(valid_proba)),
        "valid_proba_max": float(np.max(valid_proba)),
        "test_proba_min": float(np.min(test_proba)),
        "test_proba_max": float(np.max(test_proba)),
    }

    if len(np.unique(y_valid)) > 1:
        result["valid_roc_auc"] = float(roc_auc_score(y_valid, valid_proba))
        result["valid_pr_auc"] = float(average_precision_score(y_valid, valid_proba))
    else:
        result["valid_roc_auc"] = None
        result["valid_pr_auc"] = None

    if len(np.unique(y_test)) > 1:
        result["test_roc_auc"] = float(roc_auc_score(y_test, test_proba))
        result["test_pr_auc"] = float(average_precision_score(y_test, test_proba))
    else:
        result["test_roc_auc"] = None
        result["test_pr_auc"] = None

    return result


def single_feature_auc(train_df, test_df, col: str) -> dict:
    # Очень быстрый тест: если одна колонка почти идеально предсказывает target -> leakage suspect
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    series_train = train_df[col]
    series_test = test_df[col]

    if pd.api.types.is_numeric_dtype(series_train):
        train_values = series_train.fillna(series_train.median() if series_train.notna().any() else 0).values.reshape(-1, 1)
        test_values = series_test.fillna(series_train.median() if series_train.notna().any() else 0).values.reshape(-1, 1)

        model = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_iter=120,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
        )
        model.fit(train_values, y_train)
        proba = model.predict_proba(test_values)[:, 1]
    else:
        X_train = pd.DataFrame({col: series_train})
        X_test = pd.DataFrame({col: series_test})

        pipe = build_pipeline([col], [])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

    result = {
        "feature": col,
        "dtype": str(train_df[col].dtype),
        "non_null_rate_train": float(train_df[col].notna().mean()),
        "unique_train": int(train_df[col].nunique(dropna=True)),
    }

    if len(np.unique(y_test)) > 1:
        result["test_roc_auc"] = float(roc_auc_score(y_test, proba))
        result["test_pr_auc"] = float(average_precision_score(y_test, proba))
    else:
        result["test_roc_auc"] = None
        result["test_pr_auc"] = None

    return result


def compute_mutual_info(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Только числовые фичи
    use_cols = []
    for c in cols:
        if c == TARGET_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            use_cols.append(c)

    if not use_cols:
        return pd.DataFrame(columns=["feature", "mutual_info"])

    X = df[use_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    y = df[TARGET_COL].values

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)
    out = pd.DataFrame({
        "feature": use_cols,
        "mutual_info": mi,
    }).sort_values("mutual_info", ascending=False)

    return out


def write_text_report(path: str, report: dict) -> None:
    lines = []
    lines.append("LEAKAGE DEBUG REPORT")
    lines.append("=" * 80)
    lines.append(f"Dataset rows: {report['dataset']['rows']}")
    lines.append(f"Dataset cols: {report['dataset']['cols']}")
    lines.append(f"Target positive rate: {report['dataset']['target_positive_rate']:.6f}")
    lines.append("")

    lines.append("COLUMN GROUPS")
    lines.append("-" * 80)
    lines.append(f"Categorical candidates: {report['column_groups']['categorical_cols']}")
    lines.append(f"Embedding cols: {report['column_groups']['embedding_cols_count']}")
    lines.append(f"Suspicious cols count: {len(report['column_groups']['suspicious_cols'])}")
    lines.append("Top suspicious cols:")
    for c in report['column_groups']['suspicious_cols'][:50]:
        lines.append(f"  - {c}")
    lines.append("")

    lines.append("MODEL COMPARISONS")
    lines.append("-" * 80)
    for res in report["model_comparisons"]:
        lines.append(
            f"{res['name']} | status={res['status']} | "
            f"valid_roc_auc={res.get('valid_roc_auc')} | "
            f"valid_pr_auc={res.get('valid_pr_auc')} | "
            f"test_roc_auc={res.get('test_roc_auc')} | "
            f"test_pr_auc={res.get('test_pr_auc')} | "
            f"n_features={res.get('n_features')}"
        )
    lines.append("")

    lines.append("TOP SINGLE-FEATURE SUSPECTS")
    lines.append("-" * 80)
    for row in report["single_feature_top"][:30]:
        lines.append(
            f"{row['feature']} | roc_auc={row.get('test_roc_auc')} | "
            f"pr_auc={row.get('test_pr_auc')} | unique_train={row['unique_train']}"
        )
    lines.append("")

    lines.append("TOP MUTUAL INFO FEATURES")
    lines.append("-" * 80)
    for row in report["top_mutual_info"][:30]:
        lines.append(f"{row['feature']} | mi={row['mutual_info']}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_dir(OUTPUT_DIR)

    df = load_data(INPUT_CSV)
    train_df, valid_df, test_df = time_split(df)

    info = classify_columns(df)

    print("Loaded:", df.shape)
    print("Positive rate:", df[TARGET_COL].mean())
    print("Suspicious cols:", len(info["suspicious_cols"]))

    # Сохраним списки колонок
    with open(os.path.join(OUTPUT_DIR, "all_columns.json"), "w", encoding="utf-8") as f:
        json.dump(df.columns.tolist(), f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "suspicious_columns.json"), "w", encoding="utf-8") as f:
        json.dump(info["suspicious_cols"], f, ensure_ascii=False, indent=2)

    # Наборы фичей для сравнения
    strict_safe_categorical = info["strict_safe_categorical"]
    strict_safe_numeric = info["strict_safe_numeric"]

    only_embeddings = info["embedding_cols"]
    only_categorical = info["categorical_cols"]

    suspicious_numeric = [c for c in info["numeric_cols"] if c in info["suspicious_cols"]]
    suspicious_categorical = [c for c in info["categorical_cols"] if c in info["suspicious_cols"]]

    all_numeric = []
    for c in info["numeric_cols"]:
        if c in SAFE_EXACT_DROP:
            continue
        all_numeric.append(c)

    all_categorical = []
    for c in info["categorical_cols"]:
        if c in SAFE_EXACT_DROP:
            continue
        all_categorical.append(c)

    model_comparisons = []

    experiment_specs = [
        ("all_except_exact_drop", all_categorical, all_numeric),
        ("strict_safe_only", strict_safe_categorical, strict_safe_numeric),
        ("only_embeddings", [], only_embeddings),
        ("only_categorical", only_categorical, []),
        ("only_suspicious", suspicious_categorical, suspicious_numeric),
        ("only_base_safe_numeric", [], [c for c in BASE_SAFE_NUMERIC if c in df.columns]),
    ]

    for name, cat_cols, num_cols in experiment_specs:
        print(f"\nRunning experiment: {name}")
        res = evaluate_model(train_df, valid_df, test_df, cat_cols, num_cols, name)
        model_comparisons.append(res)

    # Одиночные колонки
    candidate_single_features = []

    # Сначала подозрительные
    candidate_single_features.extend(info["suspicious_cols"])

    # Потом safe numeric
    for c in strict_safe_numeric[:50]:
        if c not in candidate_single_features:
            candidate_single_features.append(c)

    # Чтобы не гонять сотни эмбеддингов, ограничим
    emb_sample = info["embedding_cols"][:20]
    for c in emb_sample:
        if c not in candidate_single_features:
            candidate_single_features.append(c)

    single_results = []
    for i, col in enumerate(candidate_single_features, start=1):
        try:
            res = single_feature_auc(train_df, test_df, col)
            single_results.append(res)
        except Exception as e:
            single_results.append({
                "feature": col,
                "error": str(e),
            })

        if i % 20 == 0 or i == len(candidate_single_features):
            print(f"Single-feature tested: {i}/{len(candidate_single_features)}")

    # Отсортируем подозреваемых
    single_results_clean = [r for r in single_results if "test_roc_auc" in r and r["test_roc_auc"] is not None]
    single_results_sorted = sorted(
        single_results_clean,
        key=lambda x: (x.get("test_roc_auc", -1), x.get("test_pr_auc", -1)),
        reverse=True
    )

    # Mutual information
    mi_df = compute_mutual_info(df, info["numeric_cols"])
    mi_df.to_csv(os.path.join(OUTPUT_DIR, "mutual_info.csv"), index=False, encoding="utf-8-sig")

    single_df = pd.DataFrame(single_results)
    single_df.to_csv(os.path.join(OUTPUT_DIR, "single_feature_results.csv"), index=False, encoding="utf-8-sig")

    report = {
        "dataset": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "target_positive_rate": float(df[TARGET_COL].mean()),
            "train_rows": int(train_df.shape[0]),
            "valid_rows": int(valid_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "train_time_min": str(train_df[TIME_COL].min()),
            "train_time_max": str(train_df[TIME_COL].max()),
            "valid_time_min": str(valid_df[TIME_COL].min()),
            "valid_time_max": str(valid_df[TIME_COL].max()),
            "test_time_min": str(test_df[TIME_COL].min()),
            "test_time_max": str(test_df[TIME_COL].max()),
        },
        "column_groups": {
            "categorical_cols": info["categorical_cols"],
            "embedding_cols_count": len(info["embedding_cols"]),
            "suspicious_cols": info["suspicious_cols"],
            "strict_safe_numeric_count": len(strict_safe_numeric),
            "strict_safe_categorical": strict_safe_categorical,
        },
        "model_comparisons": model_comparisons,
        "single_feature_top": single_results_sorted[:50],
        "top_mutual_info": mi_df.head(50).to_dict(orient="records"),
    }

    with open(os.path.join(OUTPUT_DIR, "debug_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_text_report(os.path.join(OUTPUT_DIR, "debug_report.txt"), report)

    print("\nDone.")
    print(f"Saved report: {os.path.join(OUTPUT_DIR, 'debug_report.json')}")
    print(f"Saved text report: {os.path.join(OUTPUT_DIR, 'debug_report.txt')}")
    print(f"Saved single feature results: {os.path.join(OUTPUT_DIR, 'single_feature_results.csv')}")
    print(f"Saved MI results: {os.path.join(OUTPUT_DIR, 'mutual_info.csv')}")

    print("\n=== MODEL COMPARISONS ===")
    for res in model_comparisons:
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()