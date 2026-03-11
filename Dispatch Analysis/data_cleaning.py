import pandas as pd
import numpy as np
import os

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

file_2023 = os.path.join(DATA_DIR, "2023Incidents.xlsx")
file_2024 = os.path.join(DATA_DIR, "2024Incidents.xlsx")

# ---------------------------
# Load
# ---------------------------
df_2023 = pd.read_excel(file_2023)
df_2024 = pd.read_excel(file_2024)

print("2023 shape:", df_2023.shape)
print("2024 shape:", df_2024.shape)

df_2023["year"] = 2023
df_2024["year"] = 2024

df = pd.concat([df_2023, df_2024], ignore_index=True)

print("Combined shape:", df.shape)
print(df["year"].value_counts())

print("STEP A: after concat")

# ---------------------------
# Convert timestamps
# ---------------------------
call_received_col = "Alarm DateTime"
dispatch_col = "Dispatch Time"

time_cols = [call_received_col, dispatch_col]

for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

print("\nMissing key timestamps:")
print(df[time_cols].isna().sum())

# ---------------------------
# Call processing time (seconds)
# ---------------------------
df["call_processing_seconds"] = (df[dispatch_col] - df[call_received_col]).dt.total_seconds()

print("\nCPT summary (seconds):")
print(df["call_processing_seconds"].describe())

# ---------------------------
# Clean (starter rules)
# ---------------------------
df_clean = df.dropna(subset=time_cols).copy()
df_clean = df_clean[(df_clean["call_processing_seconds"] <= 600)].copy()

print("\nRows before cleaning:", df.shape[0])
print("Rows after cleaning:", df_clean.shape[0])

# ---------------------------
# Keep Primary Fire + EMS Only (NFPA compliance dataset)
# ---------------------------
incident_col = "Incident Type"

# Keep 100, 300, 400 series (Fire, EMS, Hazard)
incident_col = "Incident Type"

mask_primary = df_clean[incident_col].str.startswith(("1", "3", "4"), na=False)
df_primary = df_clean[mask_primary].copy()

print("\nAfter keeping Fire + EMS primary series:")
print("Rows remaining:", df_primary.shape[0])

print("\nCounts by series:")
print(df_primary[incident_col].str[0].value_counts())
# ---------------------------
# Save
# ---------------------------
out_path = os.path.join(DATA_DIR, "incidents_2023_2024_cleaned.csv")
print("STEP B: about to save ->", out_path)

df_clean.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Files in data folder now:", os.listdir(DATA_DIR))
