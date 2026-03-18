import pandas as pd
import numpy as np
import os

from nlp_processor import SemanticNLPProcessor


INPUT_CSV = "Проишествия_clean.csv"
OUTPUT_CSV = "incidents_ml_ready.csv"


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )


def build_risk_score(df):
    # тяжесть
    severity_map = {
        "Легкая": 1,
        "Средняя": 2,
        "Тяжелая": 3,
        "Смертельная": 5
    }

    df["severity_weight"] = df["Тяжесть_травмы"].map(severity_map).fillna(1)

    # подрядчик
    df["contractor_flag"] = df["Подрядная_организация"].notna().astype(int)

    # стаж фактор (чем меньше стаж → выше риск)
    df["stazh_factor"] = 1 / (df["stazh_clean"] + 1)

    df["risk_score"] = (
        df["severity_weight"] +
        df["stazh_factor"] +
        df["contractor_flag"]
    )

    return df


def main():
    df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8-sig")

    print("Loaded:", df.shape)

    # --- cleaning ---
    df["stazh_clean"] = clean_numeric(df["Стаж_работы_в_организации"])
    df["age"] = 2026 - clean_numeric(df["Год_рождения"])

    # бинарные флаги
    df["is_work_time"] = df["В_рабочее_время"].astype(str).str.lower().str.contains("да").astype(int)
    df["is_workplace"] = df["На_рабочем_месте"].astype(str).str.lower().str.contains("да").astype(int)

    # --- risk ---
    df = build_risk_score(df)

    print("Risk score added")

    # --- NLP ---
    text_col = "Краткое_описание_происшествия"

    df[text_col] = df[text_col].fillna("").astype(str)

    processor = SemanticNLPProcessor()
    embeddings = processor.encode_texts(df[text_col].tolist())

    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

    # --- merge ---
    final_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # --- save ---
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_CSV)
    print(final_df.head())


if __name__ == "__main__":
    main()