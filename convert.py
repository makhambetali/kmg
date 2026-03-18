import pandas as pd

pd.read_csv("коргау_clean.csv",sep=';', encoding="utf-8-sig").to_excel("коргау_clean.xlsx", index=False)
pd.read_csv("Проишествия_clean.csv", encoding="utf-8-sig", sep=';').to_excel("Проишествия_clean.xlsx", index=False)