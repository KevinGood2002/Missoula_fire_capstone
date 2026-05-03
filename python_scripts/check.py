import pandas as pd
from pathlib import Path

### Quick Check of Processed data before merging 

base_dir = Path(__file__).resolve().parent.parent
processed = base_dir / "data" / "processed"

dispatch = pd.read_csv(processed / "all_dispatch.csv")
enroute = pd.read_csv(processed / "all_enroute.csv")
onscene = pd.read_csv(processed / "all_onscene.csv")

print("Dispatch unit_id sample:")
print(dispatch["unit_id"].dropna().value_counts().head(30))

print("\nEnroute unit_id sample:")
print(enroute["unit_id"].dropna().value_counts().head(30))

print("\nOnscene unit_id sample:")
print(onscene["unit_id"].dropna().value_counts().head(30))

print(onscene["unit_id"].value_counts().reindex(["181", "182"], fill_value=0))