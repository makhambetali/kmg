import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://csvuser:csvpass@localhost:5434/csvdb")

files = {
    "karty_korgau": "коргау.xlsx",
    "proishestviya": "Проишествия.xlsx",
}

for table_name, file_name in files.items():
    df = pd.read_excel(file_name)

    # нормализация колонок
    df.columns = [
        str(col).strip().replace("\n", " ").replace(" ", "_")
        for col in df.columns
    ]

    # 🔥 УДАЛЕНИЕ НЕНУЖНЫХ КОЛОНОК
    if table_name == "karty_korgau":
        cols_to_drop = [
            "Фото/Документы_с_места_выявления_несоответствия_или_предложения",
            "Опишите_ваше_наблюдение/предложение",
        ]

        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    print(f"\n{file_name}")
    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")

    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Written to table: {table_name}")

print("\nDone.")