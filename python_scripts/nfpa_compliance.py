import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If script is in python_scripts/ and data is a sibling folder:
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
file_path = os.path.join(DATA_DIR, "final_merged_unit_times.csv")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(file_path)

print("Loaded rows:", df.shape[0])
print("Columns:", df.columns.tolist())

# ---------------------------
# Keep Primary Fire + EMS
# ---------------------------
# New file uses "call_type" instead of "Incident Type"
incident_col = "call_type"

# Adjust keywords if needed after reviewing your data
mask_primary = df[incident_col].astype(str).str.contains(
    r"medical|fire|ems|rescue|cpr",
    case=False,
    na=False
)

df_primary = df[mask_primary].copy()

print("Rows after Fire/EMS filter:", df_primary.shape[0])

# ---------------------------
# NFPA 90% ≤ 60 sec Analysis
# ---------------------------
# New file uses "creation_to_dispatch_sec"
time_col = "creation_to_dispatch_sec"

import numpy as np

time_col = "creation_to_dispatch_sec"

# ---------------------------
# STEP 1: MACRO FILTER
# ---------------------------
df_macro = df_primary[
    (df_primary[time_col] >= 1) &
    (df_primary[time_col] <= 360)
].copy()

print("After macro filter:", df_macro.shape[0])

# ---------------------------
# STEP 2: CALCULATE STATS
# ---------------------------
mean_val = df_macro[time_col].mean()
std_val = df_macro[time_col].std()

upper_threshold = mean_val + (3 * std_val)
lower_threshold = 1

print("\nStatistical Thresholds")
print("----------------------")
print("Mean:", round(mean_val, 2))
print("Std Dev:", round(std_val, 2))
print("Upper Threshold:", round(upper_threshold, 2))

# ---------------------------
# STEP 3: SIGMA FILTER
# ---------------------------
df_final = df_macro[
    (df_macro[time_col] >= lower_threshold) &
    (df_macro[time_col] <= upper_threshold)
].copy()

# Standardize call type labels first
df_final["call_type"] = df_final["call_type"].astype(str).str.strip().str.title()

# Remove non-emergent / support call types
exclude_types = [
    "Medical Transfer",
    "Medical Standby Bls",
    "Medical Standby Als",
    "Fire Public Assist"
]

df_final = df_final[~df_final["call_type"].isin(exclude_types)].copy()

# ---------------------------
# Save filtered dataset
# ---------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

output_path = os.path.join(OUTPUT_DIR, "nfpa_filtered_calls.csv")

df_final.to_csv(output_path, index=False)

print(f"Saved filtered dataset to: {output_path}")

df_final = df_final[~df_final["call_type"].str.contains("Standby", case=False, na=False)]

print("After sigma filter:", df_final.shape[0])


# ---------------------------
# FINAL NFPA ANALYSIS
# ---------------------------
compliance_rate = (df_final[time_col] <= 60).mean()
p90 = df_final[time_col].quantile(0.90)
median = df_final[time_col].median()

print("\nNFPA Call Processing Compliance Results")
print("---------------------------------------")
print("Percent ≤ 60 sec:", round(compliance_rate * 100, 2), "%")
print("90th percentile CPT:", round(p90, 2), "seconds")
print("Median CPT:", round(median, 2), "seconds")
# ---------------------------
# Top 20 Incident Types by p90
# ---------------------------
result = (
    df_final.groupby("call_type")[time_col]
    .agg(
        count="count",
        p90=lambda x: x.quantile(0.90)
    )
    .sort_values("count", ascending=False)
    .head(20)
    .round(1)
)

print("\nTop 20 Incident Types by 90th Percentile CPT:\n")
print(result)
'''
# ---------------------------
# Visualization
# ---------------------------

top20 = result.copy()

plt.figure(figsize=(10, 8))
plt.barh(top20.index, top20["p90"])
plt.axvline(60, linestyle="--")

plt.xlabel("90th Percentile Call Processing Time (seconds)")
plt.ylabel("Call Type")
plt.title("Top 20 Call Types by 90th Percentile CPT")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

compliance = (df_final[time_col] <= 60).mean() * 100
target = 90.0

plt.figure()

plt.bar(["Your System", "NFPA Target"], [compliance, target])

plt.ylabel("Percent of Calls ≤ 60 Seconds")
plt.title("NFPA Call Processing Compliance")

plt.ylim(0, 100)

# Add labels
plt.text(0, compliance + 2, f"{compliance:.1f}%", ha='center')
plt.text(1, target + 2, f"{target:.1f}%", ha='center')

plt.tight_layout()
plt.show()


plt.figure()

plt.hist(df_final[time_col], bins=50)

plt.axvline(x=60, linestyle='--', color='#B22222')

plt.xlabel("Call Processing Time (seconds)")
plt.ylabel("Number of Calls")
plt.title("Distribution of Call Processing Time")

plt.tight_layout()
plt.show()


top20 = result.copy()

plt.figure(figsize=(10, 8))

plt.barh(top20.index, top20["p90"])

# NFPA benchmark
plt.axvline(60, linestyle="--")

plt.xlabel("90th Percentile Call Processing Time (seconds)")
plt.ylabel("Call Type")
plt.title("90th Percentile CPT by Call Type (NFPA Benchmark = 60 sec)")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
'''
import matplotlib.pyplot as plt

# ---------------------------
# Select important call types
# ---------------------------
priority_types = [
    "Medical Call Bls",
    "Medical Call Als",
    "Medical Cpr",
    "Fire Alarm",
    "Fire Structure Response",
    "Assist Fire",
    "Fire Vehicle",
    "Fire Gas Smell Inside A Building",
    "Fire Gas Line Rupture"
]

# Keep only those call types
df_priority = df_final[df_final["call_type"].isin(priority_types)].copy()

# ---------------------------
# Summarize frequency + 90th percentile
# ---------------------------
priority_result = (
    df_priority.groupby("call_type")[time_col]
    .agg(
        count="count",
        p90=lambda x: x.quantile(0.90)
    )
    .round(1)
    .sort_values("p90", ascending=True)
)

print(priority_result)

# ---------------------------
# Set colors
# ---------------------------
highlight_types = [
    "Medical Call Bls",
    "Medical Call Als",
    "Fire Structure Response"
]

colors = [
    "orange" if call_type in highlight_types else "steelblue"
    for call_type in priority_result.index
]

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(11, 7))

plt.barh(priority_result.index, priority_result["p90"], color=colors)
plt.axvline(60, color="red", linestyle="--", linewidth=2)

plt.xlabel("Seconds")
plt.ylabel("Call Processing Type")
plt.title("90th Percentile Call Processing Time for High-Priority Call Types")

# Add value labels
for i, v in enumerate(priority_result["p90"]):
    plt.text(v + 3, i, f"{v:.0f}s", va="center")

plt.tight_layout()
plt.show()