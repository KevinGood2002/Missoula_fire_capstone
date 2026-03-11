import pandas as pd
import os

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

file_path = os.path.join(DATA_DIR, "incidents_2023_2024_cleaned.csv")

# ---------------------------
# Load cleaned data
# ---------------------------
df = pd.read_csv(file_path)

print("Loaded rows:", df.shape[0])

# ---------------------------
# Keep Primary Fire + EMS
# ---------------------------
incident_col = "Incident Type"

mask_primary = df[incident_col].astype(str).str.startswith(("1", "3", "4"), na=False)
df_primary = df[mask_primary].copy()

print("Rows after Fire/EMS filter:", df_primary.shape[0])

# ---------------------------
# NFPA 90% ≤ 60 sec Analysis
# ---------------------------
compliance_rate = (
    df_primary["call_processing_seconds"] <= 60
).mean()

p90 = df_primary["call_processing_seconds"].quantile(0.90)

print("\nNFPA Call Processing Compliance Results")
print("---------------------------------------")
print("Percent ≤ 60 sec:", round(compliance_rate * 100, 2), "%")
print("90th percentile CPT:", round(p90, 2), "seconds")

# Optional: median for context
median = df_primary["call_processing_seconds"].median()
print("Median CPT:", round(median, 2), "seconds")

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

compliance = (df_primary["call_processing_seconds"] <= 60).mean() * 100
target = 90.0
'''
plt.figure()
plt.bar(["Your system", "NFPA target"], [compliance, target])
plt.ylim(0, 100)
plt.ylabel("Percent of calls ≤ 60 seconds")
plt.title("NFPA Call Processing Compliance (Fire/EMS)")
plt.show()
'''

result = (
    df_primary.groupby("Incident Type")["call_processing_seconds"]
    .agg(
        count="count",
        p90=lambda x: x.quantile(0.90)
    )
    .sort_values("count", ascending=False)
    .head(20)
    .round(1)
)

###Top 20 Visualizations

print("\nTop 20 Incident Types by 90th Percentile CPT:\n")
print(result)

# Assuming `result` already contains count + p90
top20 = result.head(20).copy()

plt.figure()

plt.barh(top20.index, top20["p90"])
plt.axvline(60, linestyle="--")  # NFPA 60-second benchmark

plt.xlabel("90th Percentile Call Processing Time (seconds)")
plt.ylabel("Incident Type")
plt.title("Top 20 Incident Types by 90th Percentile CPT")

plt.gca().invert_yaxis()  # Highest at top

plt.tight_layout()
plt.show()