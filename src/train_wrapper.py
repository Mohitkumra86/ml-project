"""Auxiliary model training wrapper for expanded dataset experiments.

Trains auxiliary classifiers on expanded preprocessed features.
IMPORTANT: This module NEVER writes to 'best_churn_model_expanded.joblib'.
Only src/tune_wrapper.py creates and saves the final tuned model artifact.
"""
import os
import time
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
)


def train_expanded_models(data: dict, model_dir: str = "models_expanded") -> pd.DataFrame:
    """Train auxiliary classifiers on expanded data.

    Saves individual auxiliary model artifacts such as 'rf_baseline_expanded.joblib'
    and 'comparison_table_expanded.csv'. Strictly never writes to
    'best_churn_model_expanded.joblib'.
    """
    os.makedirs(model_dir, exist_ok=True)

    X_train = data["X_train_processed"]
    y_train = data["y_train"]
    X_test = data["X_test_processed"]
    y_test = data["y_test"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest Baseline": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
    }

    results = []

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)

        file_slug = name.lower().replace(" ", "_") + "_expanded.joblib"
        # Explicit safeguard: ensure the forbidden filename is never written
        if file_slug == "best_churn_model_expanded.joblib":
            raise RuntimeError("train_wrapper.py attempted to write best_churn_model_expanded.joblib!")

        joblib.dump(clf, os.path.join(model_dir, file_slug))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
            "Fit Time (s)": round(fit_time, 3),
        })

    df_results = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df_results.to_csv(os.path.join(model_dir, "comparison_table_expanded.csv"), index=False)

    return df_results
