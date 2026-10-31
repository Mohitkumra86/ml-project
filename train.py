"""
Telecom Customer Churn – Model Training & Evaluation
=====================================================
Trains five classifiers on the preprocessed Telco churn data,
evaluates each on the held-out test set, prints a comparison table,
and saves trained models + the fitted preprocessor to disk via joblib.

Models
------
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Metrics
-------
Accuracy · Precision · Recall · F1-Score · ROC-AUC

Artifacts saved to  models/
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocess import load_and_preprocess

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────
# 1. Load preprocessed data
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  TELECOM CUSTOMER CHURN – MODEL TRAINING & EVALUATION")
print("=" * 70)

data = load_and_preprocess(verbose=False)

X_train = data["X_train_processed"]
X_test  = data["X_test_processed"]
y_train = data["y_train"]
y_test  = data["y_test"]

print(f"\n✅ Data loaded — Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"   Train churn rate: {y_train.mean():.4f}")
print(f"   Test  churn rate: {y_test.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────
# 2. Define models
# ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Support Vector Machine": SVC(
        kernel="rbf", probability=True, random_state=42
    ),
}

# ─────────────────────────────────────────────────────────────────────
# 3. Train, evaluate, and save each model
# ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

results = []  # collects per-model metrics for the comparison table

print("\n" + "─" * 70)

for name, model in models.items():
    print(f"\n🔧 Training: {name} …")
    t0 = time.time()

    # Train
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    results.append({
        "Model":     name,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1 Score":  f1,
        "ROC-AUC":   auc,
        "Time (s)":  round(train_time, 2),
    })

    # Print per-model report
    print(f"   ⏱  Trained in {train_time:.2f}s")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Confusion Matrix:")
    print(f"       Predicted →   No   Yes")
    print(f"       Actual No   {cm[0][0]:>5} {cm[0][1]:>5}")
    print(f"       Actual Yes  {cm[1][0]:>5} {cm[1][1]:>5}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"],
                                digits=4))

    # Save model
    safe_name = name.lower().replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
    joblib.dump(model, model_path)
    print(f"   💾 Saved → {model_path}")

# Save the fitted preprocessor (needed at inference time)
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
joblib.dump(data["preprocessor"], preprocessor_path)
print(f"\n💾 Saved fitted preprocessor → {preprocessor_path}")

# ─────────────────────────────────────────────────────────────────────
# 4. Comparison table
# ─────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

# Sort by F1 Score descending (most balanced metric for imbalanced data)
results_df = results_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("  MODEL COMPARISON TABLE  (sorted by F1 Score)")
print("=" * 70 + "\n")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# 5. Winner
# ─────────────────────────────────────────────────────────────────────
best = results_df.iloc[0]
print(f"\n🏆 Best model (by F1 Score): {best['Model']}")
print(f"   Accuracy={best['Accuracy']:.4f}  "
      f"Precision={best['Precision']:.4f}  "
      f"Recall={best['Recall']:.4f}  "
      f"F1={best['F1 Score']:.4f}  "
      f"ROC-AUC={best['ROC-AUC']:.4f}")

# Save the comparison table as CSV for later use
table_path = os.path.join(MODEL_DIR, "comparison_table.csv")
results_df.to_csv(table_path, index=False)
print(f"\n💾 Comparison table saved → {table_path}")

print("\n" + "=" * 70)
print("  ✅ Training complete — all models saved to models/")
print("=" * 70 + "\n")
