import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("incidents_ml_ready.csv")

# --- 1. Risk distribution ---
plt.figure()
df["risk_score"].hist(bins=20)
plt.title("Risk Score Distribution")
plt.xlabel("risk_score")
plt.ylabel("count")
plt.show()

# --- 2. Top organizations ---
org = (
    df.groupby("Наименование_организации_ДЗО")["risk_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
org.plot(kind="bar")
plt.title("Top Organizations by Risk")
plt.ylabel("avg risk")
plt.xticks(rotation=45)
plt.show()

# --- 3. Regions ---
region = df.groupby("Область")["risk_score"].mean()

plt.figure()
region.plot(kind="bar")
plt.title("Risk by Region")
plt.ylabel("avg risk")
plt.xticks(rotation=45)
plt.show()

# --- 4. Clusters ---
if "cluster" in df.columns:
    plt.figure()
    plt.scatter(df["pca_x"], df["pca_y"], c=df["cluster"])
    plt.title("Clusters of Incidents")
    plt.show()