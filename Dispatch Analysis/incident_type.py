# Load cleaned data
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "incidents_2023_2024_cleaned.csv"))

df.groupby("Incident Type")["call_processing_seconds"] \
  .quantile(0.90) \
  .sort_values(ascending=False) \
  .head(25)

