import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://csvuser:csvpass@localhost:5434/csvdb")

df = pd.read_csv(
    "Проишествия_clean.csv",
    encoding="utf-8-sig",
    sep=";",
)

df.to_sql("incidents", engine, if_exists="replace", index=False)

print("Done: Проишествия_clean.csv -> incidents")

df = pd.read_csv(
    "Коргау_clean.csv",
    encoding="utf-8-sig",
    sep=";",
)

df.to_sql("Karty Korgau", engine, if_exists="replace", index=False)

print("Done: Коргау_clean.csv -> Karty Korgau")