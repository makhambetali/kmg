import pandas as pd

df = pd.read_csv("коргау_clean.csv", sep=";", encoding="utf-8-sig")
print(df.shape)
print(df.columns.tolist())
print(df.head(3).to_dict(orient="records"))