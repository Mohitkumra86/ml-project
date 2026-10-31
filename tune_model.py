"""
Telecom Customer Churn – Hyperparameter Tuning & Final Model Selection
=======================================================================
Tunes the top-2 models from train.py (Logistic Regression & Random
Forest) using RandomizedSearchCV, compares tuned vs baseline, selects
the final best model, and saves it as a complete pipeline.

Outputs
-------
models/best_churn_model.pkl   – full Pipeline(preprocessor + classifier)
reports/figures/
    tuned_confusion_matrix.png
    tuned_roc_curve.png
    tuned_feature_importance.png
"""

import os
import time
import warnings
import pickle
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    make_scorer,
)

from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 1. Load preprocessed data
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  HYPERPARAMETER TUNING & FINAL MODEL SELECTION")
print("=" * 70)

data = load_and_preprocess(verbose=False)

X_train = data["X_train_processed"]
X_test  = data["X_test_processed"]
y_train = data["y_train"]
y_test  = data["y_test"]

preprocessor   = data["preprocessor"]
feature_names  = data["feature_names"]
X_train_raw    = data["X_train"]
X_test_raw     = data["X_test"]

print(f"\n✅ Data loaded — Train: {X_train.shape}  |  Test: {X_test.shape}")


def evaluate(name, model, X_ts, y_ts):
    """Evaluate a model and return a dict of metrics."""
    y_pred  = model.predict(X_ts)
    y_proba = model.predict_proba(X_ts)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_ts, y_pred),
        "Precision": precision_score(y_ts, y_pred),
        "Recall":    recall_score(y_ts, y_pred),
        "F1 Score":  f1_score(y_ts, y_pred),
        "ROC-AUC":   roc_auc_score(y_ts, y_proba),
    }


# ─────────────────────────────────────────────────────────────────────
# 2. Baseline scores (retrain quickly to have them in this script)
# ─────────────────────────────────────────────────────────────────────
print("\n📊 Training baseline models …")

lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train, y_train)
lr_base_metrics = evaluate("LR (baseline)", lr_base, X_test, y_test)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
rf_base_metrics = evaluate("RF (baseline)", rf_base, X_test, y_test)

print(f"   LR baseline  → F1={lr_base_metrics['F1 Score']:.4f}  "
      f"AUC={lr_base_metrics['ROC-AUC']:.4f}  "
      f"Recall={lr_base_metrics['Recall']:.4f}")
print(f"   RF baseline  → F1={rf_base_metrics['F1 Score']:.4f}  "
      f"AUC={rf_base_metrics['ROC-AUC']:.4f}  "
      f"Recall={rf_base_metrics['Recall']:.4f}")


# ─────────────────────────────────────────────────────────────────────
# 3. Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 3a. Logistic Regression ────────────────────────────────────────
print("\n🔧 Tuning Logistic Regression …")

lr_param_grid = {
    "C":             [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50],
    "penalty":       ["l1", "l2"],
    "solver":        ["liblinear", "saga"],
    "class_weight":  [None, "balanced",
                      {0: 1, 1: 2}, {0: 1, 1: 3}],
    "max_iter":      [1000],
}

lr_search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    param_distributions=lr_param_grid,
    n_iter=60,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
lr_search.fit(X_train, y_train)
lr_time = time.time() - t0

print(f"   ⏱  {lr_time:.1f}s  |  Best CV F1: {lr_search.best_score_:.4f}")
print(f"   Best params: {lr_search.best_params_}")

lr_tuned = lr_search.best_estimator_
lr_tuned_metrics = evaluate("LR (tuned)", lr_tuned, X_test, y_test)

# ── 3b. Random Forest ──────────────────────────────────────────────
print("\n🔧 Tuning Random Forest …")

rf_param_grid = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":     ["sqrt", "log2"],
    "class_weight":     [None, "balanced", "balanced_subsample",
                         {0: 1, 1: 2}, {0: 1, 1: 3}],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=80,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
rf_search.fit(X_train, y_train)
rf_time = time.time() - t0

print(f"   ⏱  {rf_time:.1f}s  |  Best CV F1: {rf_search.best_score_:.4f}")
print(f"   Best params: {rf_search.best_params_}")

rf_tuned = rf_search.best_estimator_
rf_tuned_metrics = evaluate("RF (tuned)", rf_tuned, X_test, y_test)


# ─────────────────────────────────────────────────────────────────────
# 4. Comparison table — baseline vs tuned
# ─────────────────────────────────────────────────────────────────────
all_metrics = [lr_base_metrics, lr_tuned_metrics,
               rf_base_metrics, rf_tuned_metrics]
comp_df = pd.DataFrame(all_metrics)

print("\n" + "=" * 70)
print("  BASELINE  vs  TUNED  COMPARISON")
print("=" * 70 + "\n")
print(comp_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# 5. Select final best model
#    Primary: F1 Score  |  Secondary: ROC-AUC  |  Tertiary: Recall
# ─────────────────────────────────────────────────────────────────────
candidates = {
    "LR (tuned)": (lr_tuned, lr_tuned_metrics),
    "RF (tuned)": (rf_tuned, rf_tuned_metrics),
}

best_name = max(candidates, key=lambda k: (
    candidates[k][1]["F1 Score"],
    candidates[k][1]["ROC-AUC"],
    candidates[k][1]["Recall"],
))
best_model, best_metrics = candidates[best_name]

print(f"\n🏆 Final best model: {best_name}")
print(f"   Accuracy  = {best_metrics['Accuracy']:.4f}")
print(f"   Precision = {best_metrics['Precision']:.4f}")
print(f"   Recall    = {best_metrics['Recall']:.4f}")
print(f"   F1 Score  = {best_metrics['F1 Score']:.4f}")
print(f"   ROC-AUC   = {best_metrics['ROC-AUC']:.4f}")

# ─────────────────────────────────────────────────────────────────────
# 6. Save complete pipeline (preprocessor + best classifier)
# ─────────────────────────────────────────────────────────────────────
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   best_model),
])

pkl_path = os.path.join(MODEL_DIR, "best_churn_model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(full_pipeline, f)
print(f"\n💾 Full pipeline saved → {pkl_path}")

# Also save with joblib for convenience
joblib_path = os.path.join(MODEL_DIR, "best_churn_model.joblib")
joblib.dump(full_pipeline, joblib_path)
print(f"💾 Full pipeline saved → {joblib_path}")

# ─────────────────────────────────────────────────────────────────────
# 7. Verification — load and predict with saved pipeline
# ─────────────────────────────────────────────────────────────────────
print("\n🔍 Verifying saved pipeline …")
with open(pkl_path, "rb") as f:
    loaded_pipeline = pickle.load(f)

# Predict on raw (un-transformed) test data to prove end-to-end works
y_verify = loaded_pipeline.predict(X_test_raw)
y_verify_proba = loaded_pipeline.predict_proba(X_test_raw)[:, 1]
verify_f1 = f1_score(y_test, y_verify)
print(f"   Loaded pipeline F1 on test set: {verify_f1:.4f}  ✅")


# ─────────────────────────────────────────────────────────────────────
# 8. Generate evaluation plots
# ─────────────────────────────────────────────────────────────────────

# ── 8a. Confusion Matrix ────────────────────────────────────────────
print("\n📊 Generating evaluation plots …")

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            annot_kws={"fontsize": 18, "fontweight": "bold"},
            linewidths=1, linecolor="white", ax=ax)
ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
ax.set_title(f"Confusion Matrix — {best_name}\n"
             f"Accuracy={best_metrics['Accuracy']:.3f}  "
             f"F1={best_metrics['F1 Score']:.3f}",
             fontsize=14, fontweight="bold")

# Add counts as percentages too
total = cm.sum()
for i in range(2):
    for j in range(2):
        pct = cm[i][j] / total * 100
        ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                ha="center", va="center", fontsize=11, color="gray")

fig.tight_layout()
cm_path = os.path.join(FIGURE_DIR, "tuned_confusion_matrix.png")
fig.savefig(cm_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"   💾 tuned_confusion_matrix.png")

# ── 8b. ROC Curve ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# Plot all four models for comparison
models_for_roc = {
    "LR (baseline)": (lr_base, "#95a5a6", "--"),
    "LR (tuned)":    (lr_tuned, "#e74c3c", "-"),
    "RF (baseline)": (rf_base, "#bdc3c7", "--"),
    "RF (tuned)":    (rf_tuned, "#3498db", "-"),
}

for label, (mdl, color, ls) in models_for_roc.items():
    prob = mdl.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    lw = 2.5 if "(tuned)" in label else 1.5
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
            label=f"{label}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves — Baseline vs Tuned Models",
             fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(alpha=0.3)

fig.tight_layout()
roc_path = os.path.join(FIGURE_DIR, "tuned_roc_curve.png")
fig.savefig(roc_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"   💾 tuned_roc_curve.png")

# ── 8c. Feature Importance (if supported) ──────────────────────────
clean_names = [n.replace("num__", "").replace("cat__", "")
               for n in feature_names]

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    imp_label = "Feature Importance (Gini)"
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
    imp_label = "Feature Importance (|Coefficient|)"
else:
    importances = None

if importances is not None:
    imp_df = (pd.DataFrame({"Feature": clean_names,
                             "Importance": importances})
              .sort_values("Importance", ascending=True)
              .tail(15))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(imp_df))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors,
            edgecolor="white", linewidth=0.8)
    ax.set_title(f"Top 15 Features — {best_name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel(imp_label)

    for i, (val, _) in enumerate(
            zip(imp_df["Importance"], imp_df["Feature"])):
        ax.text(val + importances.max() * 0.01, i, f"{val:.3f}",
                va="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fi_path = os.path.join(FIGURE_DIR, "tuned_feature_importance.png")
    fig.savefig(fi_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   💾 tuned_feature_importance.png")

# ── 8d. Classification report ──────────────────────────────────────
print(f"\n📋 Final Classification Report — {best_name}:")
print(classification_report(y_test, y_pred,
                            target_names=["No Churn", "Churn"],
                            digits=4))

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"  ✅ TUNING COMPLETE")
print(f"     Best model: {best_name}")
print(f"     Pipeline:   models/best_churn_model.pkl")
print(f"     Figures:     reports/figures/tuned_*.png")
print("=" * 70 + "\n")
