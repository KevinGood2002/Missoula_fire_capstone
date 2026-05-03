from pathlib import Path
import pandas as pd
import re

# --------------------------------------------------
# SET PATHS
# Project structure:
# Capstone/
# ├── data/
# │   ├── raw_data/
# │   └── processed/
# └── python_scripts/
#     └── combine_files.py
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "data" / "raw_data"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def clean_column_names(df):
    """Convert columns to snake_case style."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[ /-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def get_file_type(filename):
    """Identify whether a file is dispatch, enroute, or onscene."""
    name = filename.lower()

    if "disp" in name:
        return "dispatch"
    if "enrt" in name:
        return "enroute"
    if "onsc" in name:
        return "onscene"

    return None


def rename_time_column(df, file_type):
    """Rename call log datetime column based on file type."""
    possible_time_cols = [
        "call_log_date_time",
        "call_log_datetime",
    ]

    time_col_found = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col_found = col
            break

    if time_col_found is None:
        return df

    if file_type == "dispatch":
        df = df.rename(columns={time_col_found: "dispatch_time"})
    elif file_type == "enroute":
        df = df.rename(columns={time_col_found: "enroute_time"})
    elif file_type == "onscene":
        df = df.rename(columns={time_col_found: "onscene_time"})

    return df


def extract_unit_id(call_log_entry):
    """
    Convert:
    'Unit C395 Enroute Jail' -> 'C395'
    'Unit 121 Dispatch' -> '121'
    'Unit MED2 Onscene' -> 'MED2'
    """
    if pd.isna(call_log_entry):
        return pd.NA

    text = str(call_log_entry).strip()

    match = re.search(r"Unit\s+([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return pd.NA


def filter_fire_units(df):
    """
    Keep only fire units that:
    - are exactly 3 digits
    - start with 1
    - are not 181 or 182
    """
    if "unit_id" not in df.columns:
        return df

    df["unit_id"] = df["unit_id"].astype("string").str.strip()

    df = df[df["unit_id"].str.match(r"^1\d{2}$", na=False)]
    df = df[~df["unit_id"].isin(["181", "182"])]

    return df


def basic_clean(df, file_type, source_file):
    """Apply all cleaning steps for a single file."""
    df = clean_column_names(df)
    df = rename_time_column(df, file_type)

    if "call_log_entry" in df.columns:
        df["unit_id"] = df["call_log_entry"].apply(extract_unit_id)
        df = df.drop(columns=["call_log_entry"])

    df = filter_fire_units(df)

    df["source_file"] = source_file
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    return df


# --------------------------------------------------
# FIND FILES
# --------------------------------------------------
all_files = list(INPUT_FOLDER.glob("*.xlsx")) + list(INPUT_FOLDER.glob("*.xls"))

dispatch_list = []
enroute_list = []
onscene_list = []

# --------------------------------------------------
# READ, CLEAN, AND SORT FILES
# --------------------------------------------------
for file in all_files:
    file_type = get_file_type(file.name)

    if file_type is None:
        print(f"Skipping file: {file.name}")
        continue

    try:
        df = pd.read_excel(file)
        df = basic_clean(df, file_type, file.name)

        if file_type == "dispatch":
            dispatch_list.append(df)
        elif file_type == "enroute":
            enroute_list.append(df)
        elif file_type == "onscene":
            onscene_list.append(df)

        print(f"Loaded {file.name} -> {file_type} ({len(df)} rows)")

    except Exception as e:
        print(f"Error reading {file.name}: {e}")

# --------------------------------------------------
# COMBINE FILES BY TYPE
# --------------------------------------------------
dispatch_all = pd.concat(dispatch_list, ignore_index=True) if dispatch_list else pd.DataFrame()
enroute_all = pd.concat(enroute_list, ignore_index=True) if enroute_list else pd.DataFrame()
onscene_all = pd.concat(onscene_list, ignore_index=True) if onscene_list else pd.DataFrame()

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
dispatch_all.to_csv(OUTPUT_FOLDER / "all_dispatch.csv", index=False)
enroute_all.to_csv(OUTPUT_FOLDER / "all_enroute.csv", index=False)
onscene_all.to_csv(OUTPUT_FOLDER / "all_onscene.csv", index=False)

print("\nDONE")
print(f"Dispatch rows: {len(dispatch_all)}")
print(f"Enroute rows:  {len(enroute_all)}")
print(f"Onscene rows:  {len(onscene_all)}")
print(f"Saved files to: {OUTPUT_FOLDER}")