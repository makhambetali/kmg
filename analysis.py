import os
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest


INPUT_CSV = "incidents_ml_ready.csv"
OUTPUT_DIR = "outputs"

TOP_N_RISKY = 20
N_CLUSTERS = 5
TOP_N_SIMILAR = 10
ANOMALY_CONTAMINATION = 0.08
RANDOM_STATE = 42


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_")]


def safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_top_risky(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col for col in [
            "ID",
            "Наименование_организации_ДЗО",
            "Область",
            "Тяжесть_травмы",
            "Классификация_НС",
            "Краткое_описание_происшествия",
            "Предварительные_причины",
            "stazh_clean",
            "age",
            "risk_score",
        ]
        if col in df.columns
    ]

    top_risky = (
        df.sort_values("risk_score", ascending=False)
        .head(TOP_N_RISKY)
        .copy()
    )

    return top_risky[cols] if cols else top_risky


def build_org_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_col = "Наименование_организации_ДЗО"
    if group_col not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        "risk_score": ["count", "mean", "max", "median"],
    }

    if "stazh_clean" in df.columns:
        agg_dict["stazh_clean"] = ["mean", "median"]
    if "age" in df.columns:
        agg_dict["age"] = ["mean", "median"]

    org = df.groupby(group_col).agg(agg_dict)
    org.columns = ["_".join(col).strip() for col in org.columns.values]
    org = org.reset_index()

    rename_map = {
        "risk_score_count": "incident_count",
        "risk_score_mean": "avg_risk_score",
        "risk_score_max": "max_risk_score",
        "risk_score_median": "median_risk_score",
        "stazh_clean_mean": "avg_stazh",
        "stazh_clean_median": "median_stazh",
        "age_mean": "avg_age",
        "age_median": "median_age",
    }
    org = org.rename(columns=rename_map)

    org = org.sort_values(["avg_risk_score", "incident_count"], ascending=[False, False])
    return org


def build_region_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_col = "Область"
    if group_col not in df.columns:
        return pd.DataFrame()

    region = (
        df.groupby(group_col)
        .agg(
            incident_count=("risk_score", "count"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            median_risk_score=("risk_score", "median"),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "incident_count"], ascending=[False, False])
    )
    return region


def run_clustering(df: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not emb_cols:
        return df.copy(), pd.DataFrame()

    work_df = df.copy()
    X = work_df[emb_cols].fillna(0.0).values

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    work_df["cluster"] = cluster_labels

    # 2D координаты для возможной визуализации в BI
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    work_df["pca_x"] = coords[:, 0]
    work_df["pca_y"] = coords[:, 1]

    summary_rows = []
    for cluster_id in sorted(work_df["cluster"].dropna().unique()):
        cluster_df = work_df[work_df["cluster"] == cluster_id].copy()

        sample_texts = []
        if "Краткое_описание_происшествия" in cluster_df.columns:
            sample_texts = (
                safe_text(cluster_df["Краткое_описание_происшествия"])
                .loc[lambda s: s != ""]
                .head(5)
                .tolist()
            )

        top_orgs = []
        if "Наименование_организации_ДЗО" in cluster_df.columns:
            top_orgs = (
                cluster_df["Наименование_организации_ДЗО"]
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

        summary_rows.append({
            "cluster": int(cluster_id),
            "incident_count": int(len(cluster_df)),
            "avg_risk_score": round(float(cluster_df["risk_score"].mean()), 4) if "risk_score" in cluster_df.columns else np.nan,
            "top_organizations": " | ".join(map(str, top_orgs)),
            "sample_descriptions": " || ".join(map(str, sample_texts)),
        })

    cluster_summary = pd.DataFrame(summary_rows).sort_values("cluster")
    return work_df, cluster_summary


def build_similar_for_top_risky(df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    if not emb_cols or "risk_score" not in df.columns:
        return pd.DataFrame()

    work_df = df.copy().reset_index(drop=True)
    X = work_df[emb_cols].fillna(0.0).values

    top_indices = (
        work_df.sort_values("risk_score", ascending=False)
        .head(min(5, len(work_df)))
        .index
        .tolist()
    )

    rows = []
    for idx in top_indices:
        query_vec = X[idx].reshape(1, -1)
        sims = cosine_similarity(query_vec, X)[0]

        sim_df = work_df.copy()
        sim_df["similarity"] = sims
        sim_df = sim_df.drop(index=idx)
        sim_df = sim_df.sort_values("similarity", ascending=False).head(TOP_N_SIMILAR)

        for _, row in sim_df.iterrows():
            rows.append({
                "query_index": int(idx),
                "query_id": row.get("ID", ""),
                "query_org": work_df.loc[idx, "Наименование_организации_ДЗО"] if "Наименование_организации_ДЗО" in work_df.columns else "",
                "query_risk_score": work_df.loc[idx, "risk_score"],
                "query_description": work_df.loc[idx, "Краткое_описание_происшествия"] if "Краткое_описание_происшествия" in work_df.columns else "",
                "similar_incident_id": row.get("ID", ""),
                "similar_org": row.get("Наименование_организации_ДЗО", ""),
                "similar_risk_score": row.get("risk_score", np.nan),
                "similarity": round(float(row["similarity"]), 6),
                "similar_description": row.get("Краткое_описание_происшествия", ""),
            })

    return pd.DataFrame(rows)


def run_anomaly_detection(df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    feature_cols = []

    numeric_candidates = [
        "risk_score",
        "stazh_clean",
        "age",
        "is_work_time",
        "is_workplace",
        "contractor_flag",
        "severity_weight",
        "stazh_factor",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            feature_cols.append(col)

    feature_cols.extend(emb_cols)

    if not feature_cols:
        return pd.DataFrame()

    work_df = df.copy()
    X = work_df[feature_cols].fillna(0.0).values
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    work_df["is_anomaly"] = (preds == -1).astype(int)
    work_df["anomaly_score"] = scores

    anomalies = (
        work_df[work_df["is_anomaly"] == 1]
        .sort_values("anomaly_score", ascending=True)
        .copy()
    )

    cols = [
        col for col in [
            "ID",
            "Наименование_организации_ДЗО",
            "Область",
            "Тяжесть_травмы",
            "Классификация_НС",
            "Краткое_описание_происшествия",
            "risk_score",
            "anomaly_score",
        ]
        if col in anomalies.columns
    ]
    return anomalies[cols] if cols else anomalies


def write_summary_txt(
    path: str,
    df: pd.DataFrame,
    top_risky: pd.DataFrame,
    org_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    lines = []
    lines.append("ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Всего инцидентов: {len(df)}")
    lines.append(f"Всего колонок: {df.shape[1]}")
    lines.append("")

    if "risk_score" in df.columns:
        lines.append("RISK SCORE")
        lines.append("-" * 80)
        lines.append(f"Средний risk_score: {round(float(df['risk_score'].mean()), 4)}")
        lines.append(f"Максимальный risk_score: {round(float(df['risk_score'].max()), 4)}")
        lines.append(f"Минимальный risk_score: {round(float(df['risk_score'].min()), 4)}")
        lines.append("")

    if not top_risky.empty:
        lines.append("TOP RISKY INCIDENTS")
        lines.append("-" * 80)
        for i, (_, row) in enumerate(top_risky.head(5).iterrows(), start=1):
            org = row.get("Наименование_организации_ДЗО", "")
            risk = row.get("risk_score", "")
            desc = str(row.get("Краткое_описание_происшествия", ""))[:200]
            lines.append(f"{i}. {org} | risk={risk} | {desc}")
        lines.append("")

    if not org_summary.empty:
        lines.append("TOP ORGANIZATIONS BY AVG RISK")
        lines.append("-" * 80)
        for i, (_, row) in enumerate(org_summary.head(5).iterrows(), start=1):
            org = row.get("Наименование_организации_ДЗО", "")
            avg_risk = row.get("avg_risk_score", "")
            cnt = row.get("incident_count", "")
            lines.append(f"{i}. {org} | avg_risk={round(float(avg_risk), 4)} | incidents={cnt}")
        lines.append("")

    if not region_summary.empty:
        lines.append("TOP REGIONS BY AVG RISK")
        lines.append("-" * 80)
        for i, (_, row) in enumerate(region_summary.head(5).iterrows(), start=1):
            region = row.get("Область", "")
            avg_risk = row.get("avg_risk_score", "")
            cnt = row.get("incident_count", "")
            lines.append(f"{i}. {region} | avg_risk={round(float(avg_risk), 4)} | incidents={cnt}")
        lines.append("")

    if not cluster_summary.empty:
        lines.append("CLUSTER SUMMARY")
        lines.append("-" * 80)
        for _, row in cluster_summary.iterrows():
            lines.append(
                f"cluster={row['cluster']} | count={row['incident_count']} | "
                f"avg_risk={row['avg_risk_score']} | top_orgs={row['top_organizations']}"
            )
        lines.append("")

    if not anomalies.empty:
        lines.append("ANOMALIES")
        lines.append("-" * 80)
        lines.append(f"Найдено аномалий: {len(anomalies)}")
        for i, (_, row) in enumerate(anomalies.head(5).iterrows(), start=1):
            org = row.get("Наименование_организации_ДЗО", "")
            score = row.get("anomaly_score", "")
            desc = str(row.get("Краткое_описание_происшествия", ""))[:200]
            lines.append(f"{i}. {org} | anomaly_score={round(float(score), 6)} | {desc}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_metadata_json(path: str, df: pd.DataFrame, emb_cols: list[str]) -> None:
    meta = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "embedding_dimension": int(len(emb_cols)),
        "columns": df.columns.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    df = load_data(INPUT_CSV)
    emb_cols = get_embedding_columns(df)

    print(f"Loaded: {df.shape}")
    print(f"Embedding columns found: {len(emb_cols)}")

    top_risky = build_top_risky(df)
    org_summary = build_org_risk_summary(df)
    region_summary = build_region_risk_summary(df)
    clustered_df, cluster_summary = run_clustering(df, emb_cols)
    similar_df = build_similar_for_top_risky(clustered_df, emb_cols)
    anomalies = run_anomaly_detection(clustered_df, emb_cols)

    top_risky.to_csv(os.path.join(OUTPUT_DIR, "top_risky_incidents.csv"), index=False, encoding="utf-8-sig")
    org_summary.to_csv(os.path.join(OUTPUT_DIR, "org_risk_summary.csv"), index=False, encoding="utf-8-sig")
    region_summary.to_csv(os.path.join(OUTPUT_DIR, "region_risk_summary.csv"), index=False, encoding="utf-8-sig")
    clustered_df.to_csv(os.path.join(OUTPUT_DIR, "clustered_incidents.csv"), index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"), index=False, encoding="utf-8-sig")
    similar_df.to_csv(os.path.join(OUTPUT_DIR, "similar_incidents_for_top_risky.csv"), index=False, encoding="utf-8-sig")
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalous_incidents.csv"), index=False, encoding="utf-8-sig")

    write_summary_txt(
        os.path.join(OUTPUT_DIR, "analysis_summary.txt"),
        clustered_df,
        top_risky,
        org_summary,
        region_summary,
        cluster_summary,
        anomalies,
    )
    write_metadata_json(
        os.path.join(OUTPUT_DIR, "metadata.json"),
        clustered_df,
        emb_cols,
    )

    print("\nSaved files:")
    print("- outputs/top_risky_incidents.csv")
    print("- outputs/org_risk_summary.csv")
    print("- outputs/region_risk_summary.csv")
    print("- outputs/clustered_incidents.csv")
    print("- outputs/cluster_summary.csv")
    print("- outputs/similar_incidents_for_top_risky.csv")
    print("- outputs/anomalous_incidents.csv")
    print("- outputs/analysis_summary.txt")
    print("- outputs/metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    main()