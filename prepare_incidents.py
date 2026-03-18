import os
import re
import numpy as np
import pandas as pd


INPUT_CSV = "Проишествия_clean.csv"
OUTPUT_CSV = "incidents_prepared.csv"


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_yes_no(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "да", "иә", "y", "t"}
    false_values = {"false", "0", "no", "нет", "жоқ", "n", "f"}

    def convert(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, bool):
            return int(x)

        val = str(x).strip().lower()
        if val in true_values:
            return 1
        if val in false_values:
            return 0
        return np.nan

    return series.apply(convert)


def build_incident_text(row: pd.Series) -> str:
    parts = [
        f"Организация: {clean_text(row.get('Наименование_организации_ДЗО'))}",
        f"Область: {clean_text(row.get('Область'))}",
        f"Классификация НС: {clean_text(row.get('Классификация_НС'))}",
        f"Классификация ОМП: {clean_text(row.get('Классификация_ОМП'))}",
        f"Тяжесть травмы: {clean_text(row.get('Тяжесть_травмы'))}",
        f"Должность пострадавшего: {clean_text(row.get('Должность_пострадавшего'))}",
        f"Краткое описание: {clean_text(row.get('Краткое_описание_происшествия'))}",
        f"Предварительные причины: {clean_text(row.get('Предварительные_причины'))}",
        f"Корректирующие меры: {clean_text(row.get('Корректирующие_меры'))}",
        f"Рекомендации: {clean_text(row.get('Рекомендации'))}",
        f"Место происшествия: {clean_text(row.get('Место_происшествия'))}",
    ]
    return " | ".join([p for p in parts if p.strip()])


def parse_event_time(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "Время_и_дата_сообщения",
        "Точное_время_возникновения",
    ]

    parsed = None
    for col in candidates:
        if col in df.columns:
            current = pd.to_datetime(df[col], errors="coerce")
            if parsed is None:
                parsed = current
            else:
                parsed = parsed.fillna(current)

    if parsed is None:
        parsed = pd.Series([pd.NaT] * len(df))

    return parsed


def map_severity(text: str) -> float:
    text = clean_text(text).lower()

    if "смерт" in text:
        return 5.0
    if "тяж" in text:
        return 4.0
    if "сред" in text:
        return 3.0
    if "лег" in text:
        return 2.0
    if text:
        return 1.0
    return np.nan


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8-sig")

    print("Файл происшествий загружен")
    print("Размер:", df.shape)
    print("Колонки:")
    print(df.columns.tolist())

    prepared = pd.DataFrame()

    prepared["source_type"] = "incident"
    prepared["incident_id"] = np.arange(1, len(df) + 1)

    prepared["event_time"] = parse_event_time(df)
    prepared["event_date"] = prepared["event_time"].dt.date
    prepared["year"] = prepared["event_time"].dt.year
    prepared["month"] = prepared["event_time"].dt.month
    prepared["day"] = prepared["event_time"].dt.day
    prepared["hour"] = prepared["event_time"].dt.hour
    prepared["day_of_week"] = prepared["event_time"].dt.dayofweek
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype("Int64")

    prepared["organization"] = df["Наименование_организации_ДЗО"].apply(clean_text)
    prepared["region"] = df["Область"].apply(clean_text) if "Область" in df.columns else None
    prepared["location"] = df["Место_происшествия"].apply(clean_text) if "Место_происшествия" in df.columns else None

    prepared["incident_type"] = df["Классификация_НС"].apply(clean_text) if "Классификация_НС" in df.columns else ""
    prepared["incident_classification_omp"] = df["Классификация_ОМП"].apply(clean_text) if "Классификация_ОМП" in df.columns else ""
    prepared["injury_severity_raw"] = df["Тяжесть_травмы"].apply(clean_text) if "Тяжесть_травмы" in df.columns else ""
    prepared["severity_weight"] = prepared["injury_severity_raw"].apply(map_severity)

    prepared["victim_position"] = df["Должность_пострадавшего"].apply(clean_text) if "Должность_пострадавшего" in df.columns else ""
    prepared["diagnosis"] = df["Диагноз"].apply(clean_text) if "Диагноз" in df.columns else ""
    prepared["body_part"] = df["Пострадавшая_часть_тела"].apply(clean_text) if "Пострадавшая_часть_тела" in df.columns else ""

    prepared["description"] = df["Краткое_описание_происшествия"].apply(clean_text) if "Краткое_описание_происшествия" in df.columns else ""
    prepared["preliminary_causes"] = df["Предварительные_причины"].apply(clean_text) if "Предварительные_причины" in df.columns else ""
    prepared["corrective_actions"] = df["Корректирующие_меры"].apply(clean_text) if "Корректирующие_меры" in df.columns else ""
    prepared["recommendations"] = df["Рекомендации"].apply(clean_text) if "Рекомендации" in df.columns else ""

    prepared["is_work_time"] = normalize_yes_no(df["В_рабочее_время"]) if "В_рабочее_время" in df.columns else np.nan
    prepared["is_workplace"] = normalize_yes_no(df["На_рабочем_месте"]) if "На_рабочем_месте" in df.columns else np.nan

    if "Подрядная_организация" in df.columns:
        prepared["contractor_name"] = df["Подрядная_организация"].apply(clean_text)
        prepared["contractor_flag"] = prepared["contractor_name"].str.len().gt(0).astype(int)
    else:
        prepared["contractor_name"] = ""
        prepared["contractor_flag"] = 0

    if "Год_рождения" in df.columns:
        birth_year = clean_numeric(df["Год_рождения"])
        prepared["birth_year"] = birth_year
        prepared["age"] = 2026 - birth_year
    else:
        prepared["birth_year"] = np.nan
        prepared["age"] = np.nan

    if "Стаж_работы_в_организации" in df.columns:
        prepared["stazh_clean"] = clean_numeric(df["Стаж_работы_в_организации"])
    else:
        prepared["stazh_clean"] = np.nan

    prepared["incident_text"] = df.apply(build_incident_text, axis=1)

    prepared["target_incident"] = 1

    prepared = prepared.sort_values(["event_time", "organization"], na_position="last").reset_index(drop=True)

    prepared.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nГотово")
    print(f"Сохранен файл: {OUTPUT_CSV}")

    cols_to_show = [
        c for c in [
            "incident_id",
            "event_time",
            "organization",
            "region",
            "incident_type",
            "injury_severity_raw",
            "severity_weight",
            "description",
            "incident_text",
        ] if c in prepared.columns
    ]
    print("\nПример строк:")
    print(prepared[cols_to_show].head(3).to_dict(orient="records"))

    print("\nКоличество организаций:", prepared["organization"].nunique())
    print("Валидных дат:", prepared["event_time"].notna().sum())

    if "injury_severity_raw" in prepared.columns:
        print("\nСводка по тяжести:")
        print(prepared["injury_severity_raw"].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()