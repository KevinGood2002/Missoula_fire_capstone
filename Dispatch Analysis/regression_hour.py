# hour_logit_nfpa.py
# Logistic regression: probability a call meets NFPA call-processing standard (<= 60s)
# as a function of hour of day (categorical).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILE_PATH = os.path.join(DATA_DIR, "incidents_2023_2024_cleaned.csv")

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(FILE_PATH)
print("Loaded rows:", df.shape[0])

# ---------------------------
# Basic checks
# ---------------------------
required_cols = ["Alarm DateTime", "Incident Type", "call_processing_seconds"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

# ---------------------------
# Fire/EMS filter (1xx, 3xx, 4xx)
# ---------------------------
incident_col = "Incident Type"
types = df[incident_col].astype(str).str.strip()
mask_primary = types.str.startswith(("1", "3", "4"), na=False)
df_primary = df[mask_primary].copy()

print("Rows after Fire/EMS filter:", df_primary.shape[0])
print("Series counts (1/3/4):")
print(df_primary[incident_col].astype(str).str.strip().str[0].value_counts())

# ---------------------------
# Build features/target
# ---------------------------
df_primary["Alarm DateTime"] = pd.to_datetime(df_primary["Alarm DateTime"], errors="coerce")
df_primary["hour"] = df_primary["Alarm DateTime"].dt.hour

# Target: meets NFPA 60-second standard
df_primary["meets_60"] = (df_primary["call_processing_seconds"] <= 60).astype(int)

# Drop rows missing hour or CPT (should be rare)
df_hour = df_primary.dropna(subset=["hour", "call_processing_seconds"]).copy()
df_hour["hour"] = df_hour["hour"].astype(int)

print("\nRows used for hour analysis:", df_hour.shape[0])
print("Overall compliance (<=60s):", round(df_hour["meets_60"].mean() * 100, 2), "%")

# X: hour as categorical dummies (baseline = hour 0)
X = pd.get_dummies(df_hour["hour"], prefix="hour", drop_first=True)
y = df_hour["meets_60"]

# ---------------------------
# Train/test split (recommended)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# Fit logistic regression
# ---------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel evaluation (holdout test set)")
print("----------------------------------")
print("Accuracy:", round(model.score(X_test, y_test), 3))
try:
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))
except Exception as e:
    print("ROC AUC: (could not compute)", e)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

# ---------------------------
# Interpret coefficients (Odds Ratios vs hour 0)
# ---------------------------
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
odds = np.exp(coef)

print("\nHour effects vs baseline hour=0 (midnight)")
print("------------------------------------------")
out = pd.DataFrame({"log_odds_coef": coef, "odds_ratio": odds})
print(out)

print("\nMost negative (worst hours vs baseline):")
print(out.sort_values("odds_ratio").head(5))

print("\nMost positive (best hours vs baseline):")
print(out.sort_values("odds_ratio", ascending=False).head(5))

'''
# ---------------------------
# Simple visual: compliance rate by hour (raw)
# ---------------------------
hourly_compliance = df_hour.groupby("hour")["meets_60"].mean().sort_index() * 100

plt.figure()
plt.plot(hourly_compliance.index, hourly_compliance.values)
plt.axhline(90, linestyle="--")  # NFPA target line
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Percent of calls ≤ 60 sec")
plt.title("NFPA Call Processing Compliance by Hour (Fire/EMS)")
plt.xticks(range(0, 24))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
'''

'''
# ---------------------------
# Optional: Predicted probability by hour (model-based)
# ---------------------------
# Build a 24-row design matrix with the same dummy columns as X
hours = pd.DataFrame({"hour": list(range(24))})
X_hours = pd.get_dummies(hours["hour"], prefix="hour", drop_first=True)

# Ensure same columns as training X (add missing columns if any)
for col in X.columns:
    if col not in X_hours.columns:
        X_hours[col] = 0
X_hours = X_hours[X.columns]

pred_prob = model.predict_proba(X_hours)[:, 1]
pred_series = pd.Series(pred_prob * 100, index=range(24))
'''
'''
plt.figure()
plt.plot(pred_series.index, pred_series.values)
plt.axhline(90, linestyle="--")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Model-predicted % ≤ 60 sec")
plt.title("Model-Predicted NFPA Compliance by Hour (Logistic Regression)")
plt.xticks(range(0, 24))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
'''