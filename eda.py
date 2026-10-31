"""
Telecom Customer Churn – Exploratory Data Analysis & Visualizations
====================================================================
Generates publication-quality EDA plots and saves them to
reports/figures/.  Uses matplotlib + seaborn with a consistent
colour palette.

Figures produced
----------------
1.  churn_distribution.png          – target class balance
2.  churn_by_contract.png           – churn rate by contract type
3.  churn_by_internet_service.png   – churn rate by internet service
4.  churn_by_payment_method.png     – churn rate by payment method
5.  churn_vs_tenure.png             – tenure distribution by churn
6.  churn_vs_monthly_charges.png    – monthly charges by churn
7.  churn_by_senior_citizen.png     – churn rate by senior citizen
8.  correlation_heatmap.png         – numeric feature correlations
9.  churn_by_services.png           – churn rate across add-on services
10. feature_importance.png          – Random Forest feature importance
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"No": "#3498db", "Yes": "#e74c3c"}      # blue / red
PALETTE_LIST = ["#3498db", "#e74c3c"]
BAR_PALETTE = sns.color_palette("coolwarm", 6)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data",
                         "Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

# Minimal cleaning needed for plots
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📁 Figures will be saved to: {FIGURE_DIR}/\n")


def _save(fig, name):
    """Save figure and print confirmation."""
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   💾 {name}")


# ─────────────────────────────────────────────────────────────────────
# 1. Churn Distribution (count + percentage)
# ─────────────────────────────────────────────────────────────────────
print("📊 1. Churn Distribution …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
ax = axes[0]
counts = df["Churn"].value_counts()
bars = ax.bar(counts.index, counts.values, color=PALETTE_LIST,
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=13)
ax.set_title("Churn Count", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.set_xlabel("")

# Pie chart
ax = axes[1]
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
       colors=PALETTE_LIST, startangle=90, explode=(0, 0.06),
       textprops={"fontsize": 13, "fontweight": "bold"},
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Churn Percentage", fontsize=14, fontweight="bold")

fig.suptitle("Customer Churn Distribution", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_distribution.png")


# ─────────────────────────────────────────────────────────────────────
# 2. Churn by Contract Type
# ─────────────────────────────────────────────────────────────────────
print("📊 2. Churn by Contract Type …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order = ["Month-to-month", "One year", "Two year"]
ax = axes[0]
sns.countplot(data=df, x="Contract", hue="Churn", order=order,
              palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Count by Contract Type", fontweight="bold")
ax.set_xlabel("")
ax.legend(title="Churn")

ax = axes[1]
ct = df.groupby("Contract")["Churn"].apply(
    lambda s: (s == "Yes").mean() * 100).reindex(order)
bars = ax.bar(ct.index, ct.values,
              color=["#e74c3c", "#f39c12", "#2ecc71"],
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_title("Churn Rate by Contract Type", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("")
ax.set_ylim(0, ct.max() + 8)

fig.suptitle("Churn by Contract Type", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_by_contract.png")


# ─────────────────────────────────────────────────────────────────────
# 3. Churn by Internet Service
# ─────────────────────────────────────────────────────────────────────
print("📊 3. Churn by Internet Service …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

inet_order = ["DSL", "Fiber optic", "No"]
ax = axes[0]
sns.countplot(data=df, x="InternetService", hue="Churn", order=inet_order,
              palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Count by Internet Service", fontweight="bold")
ax.set_xlabel("")

ax = axes[1]
ct = df.groupby("InternetService")["Churn"].apply(
    lambda s: (s == "Yes").mean() * 100).reindex(inet_order)
bars = ax.bar(ct.index, ct.values,
              color=["#2ecc71", "#e74c3c", "#95a5a6"],
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_title("Churn Rate by Internet Service", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("")
ax.set_ylim(0, ct.max() + 8)

fig.suptitle("Churn by Internet Service Type", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_by_internet_service.png")


# ─────────────────────────────────────────────────────────────────────
# 4. Churn by Payment Method
# ─────────────────────────────────────────────────────────────────────
print("📊 4. Churn by Payment Method …")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

pay_order = df["PaymentMethod"].value_counts().index.tolist()
ax = axes[0]
sns.countplot(data=df, x="PaymentMethod", hue="Churn", order=pay_order,
              palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Count by Payment Method", fontweight="bold")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)

ax = axes[1]
ct = df.groupby("PaymentMethod")["Churn"].apply(
    lambda s: (s == "Yes").mean() * 100).reindex(pay_order)
colors = sns.color_palette("RdYlGn_r", len(ct))
bars = ax.bar(ct.index, ct.values, color=colors,
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax.set_title("Churn Rate by Payment Method", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, ct.max() + 8)

fig.suptitle("Churn by Payment Method", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_by_payment_method.png")


# ─────────────────────────────────────────────────────────────────────
# 5. Churn vs Tenure
# ─────────────────────────────────────────────────────────────────────
print("📊 5. Churn vs Tenure …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.histplot(data=df, x="tenure", hue="Churn", kde=True,
             palette=PALETTE, alpha=0.6, bins=30, ax=ax,
             edgecolor="white", linewidth=0.5)
ax.set_title("Tenure Distribution by Churn", fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Count")

ax = axes[1]
sns.boxplot(data=df, x="Churn", y="tenure", palette=PALETTE,
            width=0.5, ax=ax, flierprops=dict(marker="o", markersize=3))
ax.set_title("Tenure by Churn Status", fontweight="bold")
ax.set_ylabel("Tenure (months)")
ax.set_xlabel("Churn")

fig.suptitle("Customer Tenure vs Churn", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_vs_tenure.png")


# ─────────────────────────────────────────────────────────────────────
# 6. Churn vs Monthly Charges
# ─────────────────────────────────────────────────────────────────────
print("📊 6. Churn vs Monthly Charges …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True,
             palette=PALETTE, alpha=0.6, bins=30, ax=ax,
             edgecolor="white", linewidth=0.5)
ax.set_title("Monthly Charges Distribution", fontweight="bold")
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Count")

ax = axes[1]
sns.violinplot(data=df, x="Churn", y="MonthlyCharges", palette=PALETTE,
               inner="quartile", ax=ax)
ax.set_title("Monthly Charges by Churn", fontweight="bold")
ax.set_ylabel("Monthly Charges ($)")
ax.set_xlabel("Churn")

fig.suptitle("Monthly Charges vs Churn", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_vs_monthly_charges.png")


# ─────────────────────────────────────────────────────────────────────
# 7. Churn by Senior Citizen Status
# ─────────────────────────────────────────────────────────────────────
print("📊 7. Churn by Senior Citizen …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
sns.countplot(data=df, x="SeniorCitizen", hue="Churn",
              palette=PALETTE, edgecolor="white", ax=ax)
ax.set_title("Count by Senior Citizen Status", fontweight="bold")
ax.set_xlabel("Senior Citizen")

ax = axes[1]
ct = df.groupby("SeniorCitizen")["Churn"].apply(
    lambda s: (s == "Yes").mean() * 100)
bars = ax.bar(ct.index, ct.values, color=["#2ecc71", "#e74c3c"],
              edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13)
ax.set_title("Churn Rate: Senior vs Non-Senior", fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Senior Citizen")
ax.set_ylim(0, ct.max() + 10)

fig.suptitle("Churn by Senior Citizen Status", fontsize=16,
             fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "churn_by_senior_citizen.png")


# ─────────────────────────────────────────────────────────────────────
# 8. Correlation Heatmap (numeric features)
# ─────────────────────────────────────────────────────────────────────
print("📊 8. Correlation Heatmap …")

# Temporarily encode Churn as numeric for correlation
df_corr = df.copy()
df_corr["Churn_encoded"] = df_corr["Churn"].map({"No": 0, "Yes": 1})
df_corr["SeniorCitizen_encoded"] = df_corr["SeniorCitizen"].map(
    {"No": 0, "Yes": 1})

num_cols = ["SeniorCitizen_encoded", "tenure", "MonthlyCharges",
            "TotalCharges", "Churn_encoded"]
corr = df_corr[num_cols].corr()

# Rename for prettier labels
labels = ["Senior Citizen", "Tenure", "Monthly Charges",
          "Total Charges", "Churn"]
corr.index = labels
corr.columns = labels

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            mask=mask, square=True, linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"fontsize": 12, "fontweight": "bold"})
ax.set_title("Correlation Heatmap – Numeric Features",
             fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
_save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────
# 9. Churn Rate Across Add-on Services
# ─────────────────────────────────────────────────────────────────────
print("📊 9. Churn by Add-on Services …")

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

# Compute churn rate for "Yes" vs "No" subscribers of each service
rows = []
for col in service_cols:
    for val in ["No", "Yes"]:
        subset = df[df[col] == val]
        rate = (subset["Churn"] == "Yes").mean() * 100
        rows.append({"Service": col, "Subscribed": val, "Churn Rate (%)": rate})
svc_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=svc_df, x="Service", y="Churn Rate (%)",
            hue="Subscribed", palette={"No": "#e74c3c", "Yes": "#2ecc71"},
            edgecolor="white", ax=ax)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontweight="bold", fontsize=9,
                 padding=3)

ax.set_title("Churn Rate by Add-on Service Subscription",
             fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Churn Rate (%)")
ax.legend(title="Subscribed", loc="upper right")
ax.set_ylim(0, svc_df["Churn Rate (%)"].max() + 10)
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
_save(fig, "churn_by_services.png")


# ─────────────────────────────────────────────────────────────────────
# 10. Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────────
print("📊 10. Feature Importance (Random Forest) …")

from preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier

data = load_and_preprocess(verbose=False)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(data["X_train_processed"], data["y_train"])

importances = rf.feature_importances_
feat_names = [n.replace("num__", "").replace("cat__", "")
              for n in data["feature_names"]]
imp_df = (pd.DataFrame({"Feature": feat_names, "Importance": importances})
          .sort_values("Importance", ascending=True)
          .tail(15))  # top 15

fig, ax = plt.subplots(figsize=(10, 7))
colors = sns.color_palette("viridis", len(imp_df))
ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors,
        edgecolor="white", linewidth=0.8)
ax.set_title("Top 15 Feature Importances (Random Forest)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
ax.set_ylabel("")

# Value labels
for i, (val, name) in enumerate(
        zip(imp_df["Importance"], imp_df["Feature"])):
    ax.text(val + 0.002, i, f"{val:.3f}", va="center",
            fontweight="bold", fontsize=10)

fig.tight_layout()
_save(fig, "feature_importance.png")


# ─────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  ✅ EDA complete — 10 figures saved to {FIGURE_DIR}/")
print(f"{'=' * 60}\n")
