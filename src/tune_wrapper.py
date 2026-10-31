"""Leak-free hyperparameter tuning for the expanded dataset model.

Constructs a scikit-learn Pipeline containing the ColumnTransformer preprocessor
and RandomForestClassifier. Evaluates hyperparameters using 5-fold StratifiedKFold
cross-validation strictly on raw training data (X_train_raw, y_train), fitting
preprocessing independently inside each fold to eliminate leakage.

Evaluates on held-out test data (X_test_raw) only after tuning completes.
Sole producer of 'models_expanded/best_churn_model_expanded.joblib'.
"""
import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def tune_expanded(
    data: dict,
    model_dir: str = "models_expanded",
    n_iter: int = 15,
    random_state: int = 42,
) -> tuple:
    """Tune full Pipeline on raw training data with leak-free cross-validation.

    Parameters
    ----------
    data : dict
        Output from load_and_preprocess(..., add_status=True). Must contain
        'X_train_raw', 'y_train', 'X_test_raw', 'y_test', 'numerical_cols',
        'categorical_cols'.
    model_dir : str, default='models_expanded'
        Destination directory for tuned model artifacts.
    n_iter : int, default=15
        Number of parameter settings sampled in RandomizedSearchCV.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    best_pipeline : Pipeline
        The fitted Pipeline with steps [('preprocessor', ...), ('classifier', ...)].
    metrics : dict
        Test evaluation metrics computed on held-out X_test_raw.
    """
    os.makedirs(model_dir, exist_ok=True)

    X_train_raw = data["X_train_raw"]
    y_train = data["y_train"]
    X_test_raw = data["X_test_raw"]
    y_test = data["y_test"]
    num_cols = data["numerical_cols"]
    cat_cols = data["categorical_cols"]

    # 1. Build fresh, unfitted ColumnTransformer
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # 2. Build full pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=random_state, class_weight="balanced")),
    ])

    # 3. Hyperparameter search space
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [6, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    # 4. Cross-validation: preprocessor is fit strictly inside each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,  # Refits the full pipeline on entire X_train_raw
    )

    # Search strictly on training data
    search.fit(X_train_raw, y_train)

    best_pipeline = search.best_estimator_

    # 5. Held-out test evaluation occurs only after tuning is complete
    y_pred = best_pipeline.predict(X_test_raw)
    y_proba = best_pipeline.predict_proba(X_test_raw)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
        "Best CV ROC-AUC": float(search.best_score_),
        "Best Params": search.best_params_,
    }

    # 6. Save final tuned model artifact
    tuned_model_path = os.path.join(model_dir, "best_churn_model_expanded.joblib")
    joblib.dump(best_pipeline, tuned_model_path)

    # Also save preprocessor component
    preproc_path = os.path.join(model_dir, "preprocessor_expanded.joblib")
    joblib.dump(best_pipeline.named_steps["preprocessor"], preproc_path)

    # Save metrics summary
    metrics_summary_path = os.path.join(model_dir, "tuned_metrics_expanded.csv")
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "Best Params"}]).to_csv(
        metrics_summary_path, index=False
    )

    return best_pipeline, metrics
