# hour_logit_nfpa_clean.py
# Logistic regression on filtered NFPA-compliant dataset

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
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
FILE_PATH = os.path.join(DATA_DIR, "nfpa_filtered_calls.csv")

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(FILE_PATH)
print("Loaded rows:", df.shape[0])

# ---------------------------
# Required columns
# ---------------------------
required_cols = ["call_creation_date_and_time", "creation_to_dispatch_sec"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------------------------
# Build features
# ---------------------------
df["call_creation_date_and_time"] = pd.to_datetime(
    df["call_creation_date_and_time"], errors="coerce"
)

df["hour"] = df["call_creation_date_and_time"].dt.hour

# Target: meets NFPA standard
time_col = "creation_to_dispatch_sec"
df["meets_60"] = (df[time_col] <= 60).astype(int)

# Drop missing
df_hour = df.dropna(subset=["hour", time_col]).copy()
df_hour["hour"] = df_hour["hour"].astype(int)

print("\nRows used:", df_hour.shape[0])
print("Overall compliance:", round(df_hour["meets_60"].mean() * 100, 2), "%")

# ---------------------------
# Feature matrix
# ---------------------------
X = pd.get_dummies(df_hour["hour"], prefix="hour", drop_first=True)
y = df_hour["meets_60"]

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# Fit model
# ---------------------------
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel evaluation")
print("----------------")
print("Accuracy:", round(model.score(X_test, y_test), 3))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

# ---------------------------
# Coefficients (interpretation)
# ---------------------------
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
odds = np.exp(coef)

results = pd.DataFrame({
    "log_odds_coef": coef,
    "odds_ratio": odds
})

print("\nHour effects vs midnight baseline")
print("---------------------------------")
print(results)

print("\nWorst hours (lowest odds of meeting NFPA):")
print(results.sort_values("odds_ratio").head(5))

print("\nBest hours:")
print(results.sort_values("odds_ratio", ascending=False).head(5))

hours = list(range(24))

X_pred = pd.get_dummies(hours, prefix="hour")
X_pred = X_pred.reindex(columns=X.columns, fill_value=0)

probs = model.predict_proba(X_pred)[:, 1]

plt.figure(figsize=(10, 6))

plt.plot(hours, probs, marker="o")

# NFPA target
plt.axhline(y=0.90, linestyle="--", color="red")

plt.xlabel("Hour of Day")
plt.ylabel("Probability of Meeting NFPA (≤60 sec)")
plt.title("Probability of Meeting NFPA Standard by Hour")

plt.xticks(hours)

plt.tight_layout()
plt.show()