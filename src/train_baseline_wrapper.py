"""Baseline model training wrapper for experiments.

Trains baseline models on baseline preprocessed data and saves artifacts
to a configurable experiment directory (default: models_baseline_experiment/).
Does NOT modify or overwrite existing models/ artifacts.
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_baseline_models(data: dict, model_dir: str = "models_baseline_experiment") -> pd.DataFrame:
    """Train baseline classifiers and save to model_dir."""
    os.makedirs(model_dir, exist_ok=True)

    X_train = data["X_train_processed"]
    y_train = data["y_train"]
    X_test = data["X_test_processed"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
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

        file_slug = name.lower().replace(" ", "_")
        joblib.dump(clf, os.path.join(model_dir, f"{file_slug}.joblib"))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
            "Fit Time (s)": round(fit_time, 3),
        })

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))

    # Select best baseline model by F1 Score and build complete pipeline
    df_results = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df_results.to_csv(os.path.join(model_dir, "comparison_table.csv"), index=False)

    best_model_name = df_results.iloc[0]["Model"]
    best_clf = models[best_model_name]
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_clf),
    ])
    joblib.dump(full_pipeline, os.path.join(model_dir, "best_churn_model.joblib"))

    return df_results
