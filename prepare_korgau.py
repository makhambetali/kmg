import os
import re
import numpy as np
import pandas as pd


INPUT_CSV = "коргау_clean.csv"
OUTPUT_CSV = "korgau_prepared.csv"


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_bool(series: pd.Series) -> pd.Series:
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


def categorize_observation_type(text: str) -> str:
    text = clean_text(text).lower()

    if "опасный фактор" in text:
        return "danger_factor"
    if "небезопасное условие" in text:
        return "unsafe_condition"
    if "небезопасное действие" in text:
        return "unsafe_action"
    if "небезопасное поведение" in text:
        return "unsafe_behavior"
    if "хорошая практика" in text:
        return "good_practice"

    return "other"


def build_signal_text(row: pd.Series) -> str:
    parts = [
        f"Тип наблюдения: {clean_text(row.get('Тип_наблюдения'))}",
        f"Категория: {clean_text(row.get('Категория_наблюдения'))}",
        f"Организация: {clean_text(row.get('Организация'))}",
        f"Последствия или преимущества: {clean_text(row.get('Какие_возможные_последствия_наблюдения_или_преимущества_хорошей_практики_/_вашего_предложения?'))}",
        f"Меры: {clean_text(row.get('Какие_меры_вы_предприняли?'))}",
        f"Остановка работ: {clean_text(row.get('Производилась_ли_остановка_работ?'))}",
        f"Поощрение хорошей практики: {clean_text(row.get('Какие_действия_вы_предприняли_для_поощрения_хорошей_практики?'))}",
        f"Обсудили с наблюдаемым: {clean_text(row.get('Обсудили_ли_вы_небезопасное_действие_/_небезопасное_поведение_с_наблюдаемым?'))}",
        f"Сообщили ответственному: {clean_text(row.get('Сообщили_ли_ответственному_лицу?'))}",
        f"Кому сообщили: {clean_text(row.get('Кому_вы_сообщили_о_наблюдении?_(Обязательно_к_заполнению_для_небезопасного_действия/_небезопасного_поведения/опасного_фактора_/_опасного_случая)'))}",
        f"Опасность устранена: {clean_text(row.get('Было_ли_небезопасное_условие_/_поведение_исправлено_и_опасность_устранена?'))}",
    ]
    return " | ".join([p for p in parts if p.strip()])


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8-sig")

    print("Файл загружен")
    print("Размер:", df.shape)
    print("Колонки:")
    print(df.columns.tolist())

    prepared = pd.DataFrame()

    prepared["source_type"] = "korgau"
    prepared["korgau_id"] = np.arange(1, len(df) + 1)

    prepared["event_time"] = pd.to_datetime(df["Дата_Время"], errors="coerce")
    prepared["event_date"] = prepared["event_time"].dt.date
    prepared["year"] = prepared["event_time"].dt.year
    prepared["month"] = prepared["event_time"].dt.month
    prepared["day"] = prepared["event_time"].dt.day
    prepared["hour"] = prepared["event_time"].dt.hour
    prepared["day_of_week"] = prepared["event_time"].dt.dayofweek
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype("Int64")

    prepared["organization"] = df["Организация"].apply(clean_text)

    # Пока в исходном korgau нет явного региона/локации
    prepared["region"] = None
    prepared["location"] = None

    prepared["observer_name"] = df["ФИО"].apply(clean_text)

    prepared["observation_type_raw"] = df["Тип_наблюдения"].apply(clean_text)
    prepared["observation_type"] = prepared["observation_type_raw"].apply(categorize_observation_type)

    prepared["observation_category"] = df["Категория_наблюдения"].apply(clean_text)

    prepared["potential_consequences"] = df[
        "Какие_возможные_последствия_наблюдения_или_преимущества_хорошей_практики_/_вашего_предложения?"
    ].apply(clean_text)

    prepared["measures_taken"] = df["Какие_меры_вы_предприняли?"].apply(clean_text)

    prepared["encouragement_actions"] = df[
        "Какие_действия_вы_предприняли_для_поощрения_хорошей_практики?"
    ].apply(clean_text)

    prepared["reported_to_whom"] = df[
        "Кому_вы_сообщили_о_наблюдении?_(Обязательно_к_заполнению_для_небезопасного_действия/_небезопасного_поведения/опасного_фактора_/_опасного_случая)"
    ].apply(clean_text)

    prepared["work_stopped_flag"] = normalize_bool(df["Производилась_ли_остановка_работ?"])
    prepared["discussed_with_person_flag"] = normalize_bool(
        df["Обсудили_ли_вы_небезопасное_действие_/_небезопасное_поведение_с_наблюдаемым?"]
    )
    prepared["reported_to_responsible_flag"] = normalize_bool(df["Сообщили_ли_ответственному_лицу?"])
    prepared["hazard_eliminated_flag"] = normalize_bool(
        df["Было_ли_небезопасное_условие_/_поведение_исправлено_и_опасность_устранена?"]
    )

    prepared["signal_text"] = df.apply(build_signal_text, axis=1)

    prepared["is_risk_signal"] = prepared["observation_type"].isin(
        ["danger_factor", "unsafe_condition", "unsafe_action", "unsafe_behavior"]
    ).astype(int)

    prepared["is_good_practice"] = (prepared["observation_type"] == "good_practice").astype(int)

    prepared["has_potential_consequence_text"] = prepared["potential_consequences"].str.len().gt(0).astype(int)
    prepared["has_measures_text"] = prepared["measures_taken"].str.len().gt(0).astype(int)
    prepared["has_report_target_text"] = prepared["reported_to_whom"].str.len().gt(0).astype(int)

    prepared = prepared.sort_values(["event_time", "organization"], na_position="last").reset_index(drop=True)

    prepared.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nГотово")
    print(f"Сохранен файл: {OUTPUT_CSV}")

    print("\nПример строк:")
    print(
        prepared[
            [
                "korgau_id",
                "event_time",
                "organization",
                "observation_type",
                "observation_category",
                "is_risk_signal",
                "signal_text",
            ]
        ].head(3).to_dict(orient="records")
    )

    print("\nСводка по типам наблюдений:")
    print(prepared["observation_type"].value_counts(dropna=False))

    print("\nКоличество организаций:", prepared["organization"].nunique())
    print("Валидных дат:", prepared["event_time"].notna().sum())


if __name__ == "__main__":
    main()