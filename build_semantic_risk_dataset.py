import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


KORGAU_CSV = "korgau_with_embeddings.csv"
INCIDENTS_CSV = "incidents_with_embeddings.csv"
OUTPUT_CSV = "semantic_risk_dataset.csv"

HORIZON_7D = 7
HORIZON_14D = 14
SIMILARITY_THRESHOLD = 0.45


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def validate_required_columns(df: pd.DataFrame, df_name: str, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"В {df_name} отсутствуют колонки: {missing}")


def prepare_inputs():
    if not os.path.exists(KORGAU_CSV):
        raise FileNotFoundError(f"Файл не найден: {KORGAU_CSV}")
    if not os.path.exists(INCIDENTS_CSV):
        raise FileNotFoundError(f"Файл не найден: {INCIDENTS_CSV}")

    korgau = pd.read_csv(KORGAU_CSV, encoding="utf-8-sig")
    incidents = pd.read_csv(INCIDENTS_CSV, encoding="utf-8-sig")

    validate_required_columns(
        korgau,
        "korgau",
        ["korgau_id", "event_time", "organization", "signal_text"],
    )
    validate_required_columns(
        incidents,
        "incidents",
        ["incident_id", "event_time", "organization", "incident_text"],
    )

    korgau["event_time"] = pd.to_datetime(korgau["event_time"], errors="coerce")
    incidents["event_time"] = pd.to_datetime(incidents["event_time"], errors="coerce")

    korgau = korgau[korgau["event_time"].notna()].copy()
    incidents = incidents[incidents["event_time"].notna()].copy()

    korgau["organization"] = korgau["organization"].fillna("").astype(str).str.strip()
    incidents["organization"] = incidents["organization"].fillna("").astype(str).str.strip()

    return korgau.reset_index(drop=True), incidents.reset_index(drop=True)


def add_time_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    col = "event_time"

    out[f"{prefix}year"] = out[col].dt.year
    out[f"{prefix}month"] = out[col].dt.month
    out[f"{prefix}day"] = out[col].dt.day
    out[f"{prefix}hour"] = out[col].dt.hour
    out[f"{prefix}day_of_week"] = out[col].dt.dayofweek
    out[f"{prefix}is_weekend"] = out[col].dt.dayofweek.isin([5, 6]).astype(int)

    return out


def build_similarity_features(
    k_row: pd.Series,
    future_incidents: pd.DataFrame,
    k_emb_cols: list[str],
    i_emb_cols: list[str],
) -> dict:
    result = {
        "future_max_similarity": np.nan,
        "future_mean_similarity": np.nan,
        "future_top1_similarity": np.nan,
        "future_top3_mean_similarity": np.nan,
        "future_similar_incident_count": 0,
    }

    if future_incidents.empty:
        return result

    if not k_emb_cols or not i_emb_cols:
        return result

    k_vec = pd.DataFrame([k_row[k_emb_cols].values], columns=k_emb_cols).fillna(0.0).values
    i_matrix = future_incidents[i_emb_cols].fillna(0.0).values

    sims = cosine_similarity(k_vec, i_matrix)[0]
    sims_sorted = np.sort(sims)[::-1]

    result["future_max_similarity"] = float(np.max(sims))
    result["future_mean_similarity"] = float(np.mean(sims))
    result["future_top1_similarity"] = float(sims_sorted[0])
    result["future_top3_mean_similarity"] = float(np.mean(sims_sorted[: min(3, len(sims_sorted))]))
    result["future_similar_incident_count"] = int(np.sum(sims >= SIMILARITY_THRESHOLD))

    return result


def aggregate_future_incidents(
    future_incidents: pd.DataFrame,
    severe_threshold: float = 4.0,
) -> dict:
    result = {
        "future_incident_count": 0,
        "future_severe_incident_count": 0,
        "future_max_severity": np.nan,
        "future_mean_severity": np.nan,
        "future_min_days_to_incident": np.nan,
        "future_mean_days_to_incident": np.nan,
    }

    if future_incidents.empty:
        return result

    result["future_incident_count"] = int(len(future_incidents))

    if "severity_weight" in future_incidents.columns:
        sev = pd.to_numeric(future_incidents["severity_weight"], errors="coerce")
        result["future_severe_incident_count"] = int((sev >= severe_threshold).sum())
        result["future_max_severity"] = float(sev.max()) if sev.notna().any() else np.nan
        result["future_mean_severity"] = float(sev.mean()) if sev.notna().any() else np.nan

    if "days_to_incident" in future_incidents.columns:
        days = pd.to_numeric(future_incidents["days_to_incident"], errors="coerce")
        result["future_min_days_to_incident"] = float(days.min()) if days.notna().any() else np.nan
        result["future_mean_days_to_incident"] = float(days.mean()) if days.notna().any() else np.nan

    return result


def count_past_korgau_signals(
    korgau: pd.DataFrame,
    current_row: pd.Series,
    days_back: int,
) -> int:
    org = current_row["organization"]
    current_time = current_row["event_time"]
    start_time = current_time - pd.Timedelta(days=days_back)

    mask = (
        (korgau["organization"] == org) &
        (korgau["event_time"] < current_time) &
        (korgau["event_time"] >= start_time)
    )
    return int(mask.sum())


def count_past_risk_signals(
    korgau: pd.DataFrame,
    current_row: pd.Series,
    days_back: int,
) -> int:
    if "is_risk_signal" not in korgau.columns:
        return 0

    org = current_row["organization"]
    current_time = current_row["event_time"]
    start_time = current_time - pd.Timedelta(days=days_back)

    mask = (
        (korgau["organization"] == org) &
        (korgau["event_time"] < current_time) &
        (korgau["event_time"] >= start_time) &
        (korgau["is_risk_signal"] == 1)
    )
    return int(mask.sum())


def build_dataset():
    korgau, incidents = prepare_inputs()

    korgau = add_time_features(korgau)
    incidents = add_time_features(incidents)

    k_emb_cols = get_embedding_columns(korgau)
    i_emb_cols = get_embedding_columns(incidents)

    print(f"Korgau shape: {korgau.shape}")
    print(f"Incidents shape: {incidents.shape}")
    print(f"Korgau embedding dims: {len(k_emb_cols)}")
    print(f"Incident embedding dims: {len(i_emb_cols)}")

    rows = []

    incidents_by_org = {
        org: group.sort_values("event_time").copy()
        for org, group in incidents.groupby("organization", dropna=False)
    }

    for idx, k_row in korgau.iterrows():
        org = k_row["organization"]
        k_time = k_row["event_time"]

        org_incidents = incidents_by_org.get(org, pd.DataFrame()).copy()

        if org_incidents.empty:
            future_7d = pd.DataFrame()
            future_14d = pd.DataFrame()
        else:
            org_incidents["days_to_incident"] = (
                org_incidents["event_time"] - k_time
            ).dt.total_seconds() / (24 * 3600)

            future_7d = org_incidents[
                (org_incidents["event_time"] > k_time) &
                (org_incidents["event_time"] <= k_time + pd.Timedelta(days=HORIZON_7D))
            ].copy()

            future_14d = org_incidents[
                (org_incidents["event_time"] > k_time) &
                (org_incidents["event_time"] <= k_time + pd.Timedelta(days=HORIZON_14D))
            ].copy()

        row = {
            "korgau_id": k_row.get("korgau_id"),
            "event_time": k_time,
            "organization": org,
            "region": k_row.get("region", None),
            "location": k_row.get("location", None),
            "observer_name": k_row.get("observer_name", ""),
            "observation_type": k_row.get("observation_type", ""),
            "observation_category": k_row.get("observation_category", ""),
            "signal_text": k_row.get("signal_text", ""),
            "is_risk_signal": k_row.get("is_risk_signal", np.nan),
            "is_good_practice": k_row.get("is_good_practice", np.nan),
            "work_stopped_flag": k_row.get("work_stopped_flag", np.nan),
            "reported_to_responsible_flag": k_row.get("reported_to_responsible_flag", np.nan),
            "hazard_eliminated_flag": k_row.get("hazard_eliminated_flag", np.nan),
            "year": k_row.get("year", np.nan),
            "month": k_row.get("month", np.nan),
            "day": k_row.get("day", np.nan),
            "hour": k_row.get("hour", np.nan),
            "day_of_week": k_row.get("day_of_week", np.nan),
            "is_weekend": k_row.get("is_weekend", np.nan),
        }

        # past-history features
        row["past_korgau_count_1d"] = count_past_korgau_signals(korgau, k_row, 1)
        row["past_korgau_count_7d"] = count_past_korgau_signals(korgau, k_row, 7)
        row["past_korgau_count_14d"] = count_past_korgau_signals(korgau, k_row, 14)
        row["past_risk_signal_count_7d"] = count_past_risk_signals(korgau, k_row, 7)
        row["past_risk_signal_count_14d"] = count_past_risk_signals(korgau, k_row, 14)

        # future aggregates for 7d
        future_7d_agg = aggregate_future_incidents(future_7d)
        row["future_incident_count_7d"] = future_7d_agg["future_incident_count"]
        row["future_severe_incident_count_7d"] = future_7d_agg["future_severe_incident_count"]
        row["future_max_severity_7d"] = future_7d_agg["future_max_severity"]
        row["future_mean_severity_7d"] = future_7d_agg["future_mean_severity"]
        row["future_min_days_to_incident_7d"] = future_7d_agg["future_min_days_to_incident"]
        row["future_mean_days_to_incident_7d"] = future_7d_agg["future_mean_days_to_incident"]

        # future aggregates for 14d
        future_14d_agg = aggregate_future_incidents(future_14d)
        row["future_incident_count_14d"] = future_14d_agg["future_incident_count"]
        row["future_severe_incident_count_14d"] = future_14d_agg["future_severe_incident_count"]
        row["future_max_severity_14d"] = future_14d_agg["future_max_severity"]
        row["future_mean_severity_14d"] = future_14d_agg["future_mean_severity"]
        row["future_min_days_to_incident_14d"] = future_14d_agg["future_min_days_to_incident"]
        row["future_mean_days_to_incident_14d"] = future_14d_agg["future_mean_days_to_incident"]

        # similarity features
        sim_7d = build_similarity_features(k_row, future_7d, k_emb_cols, i_emb_cols)
        row["future_max_similarity_7d"] = sim_7d["future_max_similarity"]
        row["future_mean_similarity_7d"] = sim_7d["future_mean_similarity"]
        row["future_top1_similarity_7d"] = sim_7d["future_top1_similarity"]
        row["future_top3_mean_similarity_7d"] = sim_7d["future_top3_mean_similarity"]
        row["future_similar_incident_count_7d"] = sim_7d["future_similar_incident_count"]

        sim_14d = build_similarity_features(k_row, future_14d, k_emb_cols, i_emb_cols)
        row["future_max_similarity_14d"] = sim_14d["future_max_similarity"]
        row["future_mean_similarity_14d"] = sim_14d["future_mean_similarity"]
        row["future_top1_similarity_14d"] = sim_14d["future_top1_similarity"]
        row["future_top3_mean_similarity_14d"] = sim_14d["future_top3_mean_similarity"]
        row["future_similar_incident_count_14d"] = sim_14d["future_similar_incident_count"]

        # targets
        row["incident_in_next_7d"] = int(row["future_incident_count_7d"] > 0)
        row["incident_in_next_14d"] = int(row["future_incident_count_14d"] > 0)
        row["severe_incident_in_next_14d"] = int(row["future_severe_incident_count_14d"] > 0)
        row["semantically_similar_incident_in_next_14d"] = int(
            row["future_similar_incident_count_14d"] > 0
        )

        # add korgau embeddings as model features
        for col in k_emb_cols:
            row[col] = k_row.get(col, np.nan)

        rows.append(row)

        if (idx + 1) % 100 == 0 or idx == len(korgau) - 1:
            print(f"Processed {idx + 1}/{len(korgau)}")

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values(["event_time", "organization"], na_position="last").reset_index(drop=True)

    dataset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nГотово.")
    print(f"Сохранен файл: {OUTPUT_CSV}")
    print("Размер:", dataset.shape)

    target_cols = [
        "incident_in_next_7d",
        "incident_in_next_14d",
        "severe_incident_in_next_14d",
        "semantically_similar_incident_in_next_14d",
    ]

    print("\nСводка по target:")
    for col in target_cols:
        if col in dataset.columns:
            print(f"{col}:")
            print(dataset[col].value_counts(dropna=False))
            print()

    return dataset


if __name__ == "__main__":
    build_dataset()