from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"

dispatch = pd.read_csv(PROCESSED / "all_dispatch.csv")
enroute = pd.read_csv(PROCESSED / "all_enroute.csv")
onscene = pd.read_csv(PROCESSED / "all_onscene.csv")

# --------------------------------------------------
# CREATE RESPONSE ID
# --------------------------------------------------
def create_response_id(df):
    df["incident_number"] = df["incident_number"].astype(str).str.strip()
    df["call_number"] = df["call_number"].astype(str).str.strip()
    df["unit_id"] = df["unit_id"].astype(str).str.strip()

    df["response_id"] = (
        df["incident_number"] + "_" +
        df["call_number"] + "_" +
        df["unit_id"]
    )

    return df

dispatch = create_response_id(dispatch)
enroute = create_response_id(enroute)
onscene = create_response_id(onscene)

# --------------------------------------------------
# PARSE TIMES
# --------------------------------------------------
for df in [dispatch, enroute, onscene]:
    for col in ["dispatch_time", "enroute_time", "onscene_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# --------------------------------------------------
# HANDLE DUPLICATES (VERY IMPORTANT)
# Keep earliest time per response
# --------------------------------------------------
dispatch = (
    dispatch.sort_values("dispatch_time")
    .drop_duplicates("response_id", keep="first")
)

enroute = (
    enroute.sort_values("enroute_time")
    .drop_duplicates("response_id", keep="first")
)

onscene = (
    onscene.sort_values("onscene_time")
    .drop_duplicates("response_id", keep="first")
)

# --------------------------------------------------
# KEEP ONLY NEEDED COLUMNS
# --------------------------------------------------
dispatch = dispatch[[
    "response_id",
    "incident_number",
    "call_number",
    "unit_id",
    "dispatch_time",
    "call_creation_date_and_time",
    "call_type",
    "call_current_address"
]]

enroute = enroute[[
    "response_id",
    "enroute_time"
]]

onscene = onscene[[
    "response_id",
    "onscene_time"
]]

# --------------------------------------------------
# MERGE
# --------------------------------------------------
merged = dispatch.merge(enroute, on="response_id", how="left")
merged = merged.merge(onscene, on="response_id", how="left")

# --------------------------------------------------
# CREATE TIME METRICS
# --------------------------------------------------
merged["dispatch_to_enroute_sec"] = (
    (merged["enroute_time"] - merged["dispatch_time"])
    .dt.total_seconds()
)

merged["enroute_to_onscene_sec"] = (
    (merged["onscene_time"] - merged["enroute_time"])
    .dt.total_seconds()
)

merged["creation_to_dispatch_sec"] = (
    (
        merged["dispatch_time"] -
        pd.to_datetime(merged["call_creation_date_and_time"], errors="coerce")
    ).dt.total_seconds()
)

# --------------------------------------------------
# SAVE FINAL DATASET
# --------------------------------------------------
OUTPUT_PATH = PROCESSED / "final_merged_unit_times.csv"
merged.to_csv(OUTPUT_PATH, index=False)

print("\nDONE")
print(f"Final rows: {len(merged)}")
print(f"Saved to: {OUTPUT_PATH}")