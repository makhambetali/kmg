import os
import json
import pandas as pd
import numpy as np

INPUT_CSV = "semantic_risk_dataset.csv"
SINGLE_FEATURE_CSV = "leakage_debug/single_feature_results.csv"
OUTPUT_DIR = "problem_structure_debug"
TARGET_COL = "incident_in_next_14d"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[df["event_time"].notna()].copy()
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def top_single_feature_leaks():
    if not os.path.exists(SINGLE_FEATURE_CSV):
        return pd.DataFrame()

    df = pd.read_csv(SINGLE_FEATURE_CSV, encoding="utf-8-sig")
    if "test_roc_auc" in df.columns:
        df = df.sort_values(["test_roc_auc", "test_pr_auc"], ascending=[False, False])
    return df


def analyze_target_by_organization(df: pd.DataFrame):
    if "organization" not in df.columns:
        return pd.DataFrame()

    out = (
        df.groupby("organization")
        .agg(
            rows=(TARGET_COL, "count"),
            positives=(TARGET_COL, "sum"),
            positive_rate=(TARGET_COL, "mean"),
            min_time=("event_time", "min"),
            max_time=("event_time", "max"),
        )
        .reset_index()
        .sort_values(["positive_rate", "positives", "rows"], ascending=[False, False, False])
    )
    return out


def analyze_target_by_category(df: pd.DataFrame):
    cols = [c for c in ["observation_type", "observation_category"] if c in df.columns]
    if not cols:
        return pd.DataFrame()

    out = (
        df.groupby(cols)
        .agg(
            rows=(TARGET_COL, "count"),
            positives=(TARGET_COL, "sum"),
            positive_rate=(TARGET_COL, "mean"),
        )
        .reset_index()
        .sort_values(["positive_rate", "positives", "rows"], ascending=[False, False, False])
    )
    return out


def analyze_target_by_org_and_category(df: pd.DataFrame):
    cols = [c for c in ["organization", "observation_type", "observation_category"] if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()

    out = (
        df.groupby(cols)
        .agg(
            rows=(TARGET_COL, "count"),
            positives=(TARGET_COL, "sum"),
            positive_rate=(TARGET_COL, "mean"),
            min_time=("event_time", "min"),
            max_time=("event_time", "max"),
        )
        .reset_index()
        .sort_values(["positive_rate", "positives", "rows"], ascending=[False, False, False])
    )
    return out


def analyze_duplicates(df: pd.DataFrame):
    subset_cols = [c for c in ["organization", "observation_type", "observation_category", "signal_text"] if c in df.columns]
    if not subset_cols:
        return pd.DataFrame(), 0

    dup_mask = df.duplicated(subset=subset_cols, keep=False)
    dups = df.loc[dup_mask, subset_cols + [TARGET_COL, "event_time"]].copy()
    return dups, int(dup_mask.sum())


def organization_overlap_report(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    train_orgs = set(train_df["organization"].dropna().astype(str))
    valid_orgs = set(valid_df["organization"].dropna().astype(str))
    test_orgs = set(test_df["organization"].dropna().astype(str))

    report = {
        "train_org_count": len(train_orgs),
        "valid_org_count": len(valid_orgs),
        "test_org_count": len(test_orgs),
        "valid_seen_in_train": len(valid_orgs & train_orgs),
        "test_seen_in_train": len(test_orgs & train_orgs),
        "valid_unseen": len(valid_orgs - train_orgs),
        "test_unseen": len(test_orgs - train_orgs),
        "sample_test_unseen": sorted(list(test_orgs - train_orgs))[:20],
    }
    return report


def positive_rows_inspection(df: pd.DataFrame):
    cols = [
        c for c in [
            "event_time",
            "organization",
            "observation_type",
            "observation_category",
            "signal_text",
            TARGET_COL,
        ] if c in df.columns
    ]

    pos = df[df[TARGET_COL] == 1].copy().sort_values("event_time")
    return pos[cols]


def main():
    ensure_dir(OUTPUT_DIR)
    df = load_data()

    single_feature_top = top_single_feature_leaks()
    org_stats = analyze_target_by_organization(df)
    category_stats = analyze_target_by_category(df)
    org_cat_stats = analyze_target_by_org_and_category(df)
    dups_df, dup_count = analyze_duplicates(df)
    overlap = organization_overlap_report(df)
    pos_rows = positive_rows_inspection(df)

    if not single_feature_top.empty:
        save_df(single_feature_top.head(50), os.path.join(OUTPUT_DIR, "top_single_feature_leaks.csv"))

    if not org_stats.empty:
        save_df(org_stats, os.path.join(OUTPUT_DIR, "target_by_organization.csv"))

    if not category_stats.empty:
        save_df(category_stats, os.path.join(OUTPUT_DIR, "target_by_category.csv"))

    if not org_cat_stats.empty:
        save_df(org_cat_stats.head(500), os.path.join(OUTPUT_DIR, "target_by_org_and_category.csv"))

    if not dups_df.empty:
        save_df(dups_df, os.path.join(OUTPUT_DIR, "possible_duplicates.csv"))

    save_df(pos_rows, os.path.join(OUTPUT_DIR, "positive_rows.csv"))

    report = {
        "dataset_rows": int(len(df)),
        "positive_count": int(df[TARGET_COL].sum()),
        "positive_rate": float(df[TARGET_COL].mean()),
        "duplicate_rows_on_key_fields": dup_count,
        "organization_overlap": overlap,
        "top_org_positive_rates": org_stats.head(20).to_dict(orient="records") if not org_stats.empty else [],
        "top_category_positive_rates": category_stats.head(20).to_dict(orient="records") if not category_stats.empty else [],
        "top_org_category_positive_rates": org_cat_stats.head(20).to_dict(orient="records") if not org_cat_stats.empty else [],
    }

    with open(os.path.join(OUTPUT_DIR, "structure_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("Done.")
    print("Saved:")
    print("- problem_structure_debug/structure_report.json")
    print("- problem_structure_debug/top_single_feature_leaks.csv")
    print("- problem_structure_debug/target_by_organization.csv")
    print("- problem_structure_debug/target_by_category.csv")
    print("- problem_structure_debug/target_by_org_and_category.csv")
    print("- problem_structure_debug/possible_duplicates.csv")
    print("- problem_structure_debug/positive_rows.csv")


if __name__ == "__main__":
    main()