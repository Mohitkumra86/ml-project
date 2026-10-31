"""
Telecom Customer Churn – Exploratory Data Analysis
====================================================
This script performs initial data inspection on the Telco-Customer-Churn
dataset and prints a concise summary of data-quality issues.
"""

import pandas as pd
import os

# ── 1. Load the dataset ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("  TELECOM CUSTOMER CHURN – DATA ANALYSIS REPORT")
print("=" * 70)

# ── 2. Shape (rows & columns) ───────────────────────────────────────────────
print(f"\n📐 Dataset Shape: {df.shape[0]} rows  ×  {df.shape[1]} columns\n")

# ── 3. Column names ─────────────────────────────────────────────────────────
print("📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:>2}. {col}")

# ── 4. Data types ────────────────────────────────────────────────────────────
print("\n🔤 Data Types:")
print(df.dtypes.to_string())

# ── 5. Missing values ───────────────────────────────────────────────────────
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"\n❓ Missing Values (total = {total_missing}):")
if total_missing == 0:
    print("   No null values detected (but see data-quality notes below).")
else:
    print(missing[missing > 0].to_string())

# ── 6. Duplicate rows ───────────────────────────────────────────────────────
dup_count = df.duplicated().sum()
print(f"\n🔁 Duplicate Rows: {dup_count}")

# ── 7. Target column distribution ───────────────────────────────────────────
print("\n🎯 Target Column – 'Churn' Distribution:")
churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100
for label in churn_counts.index:
    print(f"   {label:>4}:  {churn_counts[label]:>5}  ({churn_pct[label]:.2f}%)")

# ── 8. Data-quality deep dive ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("  DATA-QUALITY ISSUES & PREPROCESSING NOTES")
print("=" * 70)

issues_found = 0

# 8a. TotalCharges should be numeric but often contains whitespace strings
print("\n🔍 [Check] TotalCharges column …")
tc_non_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad_tc = tc_non_numeric.isna().sum() - df["TotalCharges"].isna().sum()  # truly non-numeric
if bad_tc > 0:
    issues_found += 1
    print(f"   ⚠️  {bad_tc} rows have non-numeric (likely blank/whitespace) TotalCharges.")
    print("       These will need to be converted or imputed during preprocessing.")
    # Show sample bad rows
    mask = tc_non_numeric.isna() & df["TotalCharges"].notna()
    sample = df.loc[mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].head()
    print(sample.to_string(index=False))
else:
    print("   ✅ TotalCharges is clean – all values are numeric.")

# 8b. SeniorCitizen encoding (0/1 instead of Yes/No like other binary cols)
print("\n🔍 [Check] SeniorCitizen encoding …")
sc_unique = sorted(df["SeniorCitizen"].unique())
print(f"   Unique values: {sc_unique}")
if set(sc_unique) == {0, 1}:
    issues_found += 1
    print("   ⚠️  Encoded as 0/1 while other binary columns use Yes/No.")
    print("       Consider making encoding consistent during preprocessing.")

# 8c. customerID – not useful for modelling
print("\n🔍 [Check] customerID column …")
unique_ids = df["customerID"].nunique()
print(f"   Unique IDs: {unique_ids} / {len(df)} rows")
if unique_ids == len(df):
    issues_found += 1
    print("   ⚠️  Every row has a unique ID → drop before training (no predictive value).")

# 8d. Class imbalance assessment
print("\n🔍 [Check] Class imbalance …")
minority_pct = churn_pct.min()
if minority_pct < 30:
    issues_found += 1
    print(f"   ⚠️  Minority class is {minority_pct:.1f}% – moderate imbalance.")
    print("       Consider SMOTE, class weights, or stratified splitting.")
else:
    print(f"   ✅ Classes are reasonably balanced ({minority_pct:.1f}% minority).")

# 8e. Columns with "No internet service" / "No phone service" values
print("\n🔍 [Check] Redundant category levels …")
redundant_cols = []
for col in df.select_dtypes(include=["object", "str"]).columns:
    vals = df[col].unique()
    if "No internet service" in vals or "No phone service" in vals:
        redundant_cols.append(col)
if redundant_cols:
    issues_found += 1
    print(f"   ⚠️  {len(redundant_cols)} columns contain 'No internet/phone service':")
    for c in redundant_cols:
        print(f"       – {c}: {sorted(df[c].unique())}")
    print("       Consider collapsing these into 'No' during preprocessing.")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  ✅ Analysis complete  |  {issues_found} data-quality issue(s) flagged")
print("=" * 70 + "\n")
