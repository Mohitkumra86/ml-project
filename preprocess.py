"""
Telecom Customer Churn – Data Preprocessing Pipeline
=====================================================
Loads the raw Telco-Customer-Churn CSV, cleans it, builds a
scikit-learn ColumnTransformer pipeline, and produces
train / test splits ready for modelling.

Key design decisions
--------------------
* Pipeline is fitted **only on training data** → no data leakage.
* Stratified split preserves the ~26.5 / 73.5 % class ratio.
* TotalCharges blanks (11 rows, all tenure == 0) are coerced to NaN
  and imputed with the median inside the pipeline.
* OneHotEncoder uses handle_unknown="ignore" so unseen categories at
  inference time won't crash the pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from typing import Optional

FORBIDDEN_FEATURES = {
    "Churn", "Churn Label", "Churn Value", "Churn Score",
    "CLTV", "Churn Reason", "Churn Category", "Customer Status",
    "customerID", "Customer ID", "Status ID", "Count", "Quarter"
}


def load_and_preprocess(
    verbose: bool = False,
    add_status: bool = False,
    raw_df: Optional[pd.DataFrame] = None,
    forced_indices: Optional[dict] = None,
):
    """
    Load the Telco churn CSV, clean it, build a preprocessing
    pipeline, split into train/test, and return everything needed
    for model training.

    Parameters
    ----------
    verbose : bool
        If True, print progress messages.
    add_status : bool, default=False
        If True, merge with Telco_customer_churn_status.xlsx and include
        only 'Satisfaction Score' as an experimental feature.
    raw_df : pd.DataFrame, optional
        Pre-loaded raw DataFrame. If None, reads from Telco-Customer-Churn.csv.
    forced_indices : dict, optional
        Dict with 'train_idx' and 'test_idx' to enforce identical splits across
        experiments.

    Returns
    -------
    dict with keys:
        X_train, X_test           – raw feature DataFrames (before pipeline)
        y_train, y_test           – target Series (0/1)
        X_train_processed         – numpy array, pipeline-transformed training features
        X_test_processed          – numpy array, pipeline-transformed test features
        preprocessor              – fitted ColumnTransformer
        feature_names             – array of engineered feature names
        numerical_cols            – list of numerical column names
        categorical_cols          – list of categorical column names
        X_train_raw, X_test_raw   – explicit aliases for raw feature DataFrames
        train_customer_ids        – Series of customer IDs for training set
        test_customer_ids         – Series of customer IDs for test set
    """
    _p = print if verbose else (lambda *a, **k: None)

    # ── 1. Load the dataset ─────────────────────────────────────────
    if raw_df is not None:
        df = raw_df.copy()
        _p(f"✅ Using provided dataframe: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        DATA_PATH = os.path.join(os.path.dirname(__file__), "data",
                                 "Telco-Customer-Churn.csv")
        df = pd.read_csv(DATA_PATH)
        _p(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Optional: Merge Telco_customer_churn_status.xlsx ───────────
    if add_status:
        STATUS_PATH = os.path.join(os.path.dirname(__file__), "data",
                                   "Telco_customer_churn_status.xlsx")
        status_df = pd.read_excel(STATUS_PATH)
        status_subset = status_df[["Customer ID", "Satisfaction Score"]].copy()
        merged = pd.merge(
            df,
            status_subset,
            left_on="customerID",
            right_on="Customer ID",
            how="inner",
            validate="one_to_one",
        )
        assert len(merged) == 7043, f"Expected 7043 rows, got {len(merged)}"
        assert merged["customerID"].nunique() == 7043, "Duplicate customer IDs found after merge!"
        assert merged["Satisfaction Score"].isna().sum() == 0, "Missing Satisfaction Score found!"
        if "Customer ID" in merged.columns:
            merged = merged.drop(columns=["Customer ID"])
        df = merged
        _p("✅ Merged with Telco_customer_churn_status.xlsx (+Satisfaction Score)")

    # ── 2. Preserve customer IDs separately ────────────────────────
    customer_ids = df["customerID"].copy()

    # ── 3. Convert TotalCharges to numeric ──────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isna().sum()
    _p(f"✅ Converted 'TotalCharges' to float64  "
       f"({nan_count} NaN from blank strings)")

    # ── 4. Handle missing values ────────────────────────────────────
    total_missing = df.isna().sum().sum()
    _p(f"✅ Total missing values in dataframe: {total_missing}  "
       f"(all in TotalCharges — will be imputed inside the pipeline)")

    # ── 5. Encode target column:  Churn → 0 / 1 ────────────────────
    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["Churn"] = df["Churn"].astype(int)
    assert df["Churn"].isna().sum() == 0, "Unexpected values in Churn column!"
    _p("✅ Encoded 'Churn':  No → 0 | Yes → 1")

    # ── 6. Separate features (X) and target (y) ────────────────────
    y = df["Churn"].copy()
    cols_to_drop = [c for c in df.columns if c in FORBIDDEN_FEATURES]
    X = df.drop(columns=cols_to_drop)

    # Assert strict leakage prevention
    leakage = FORBIDDEN_FEATURES.intersection(set(X.columns))
    assert not leakage, f"Forbidden leakage features found in X: {leakage}"
    if add_status:
        assert "Satisfaction Score" in X.columns, "Satisfaction Score missing from X!"

    _p(f"✅ Features X: {X.shape}  |  Target y: {y.shape}")

    # ── 7. Identify numerical vs. categorical columns ───────────────
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "str", "string"]).columns.tolist()

    _p(f"\n📊 Numerical columns  ({len(numerical_cols)}):")
    for c in numerical_cols:
        _p(f"   • {c}")

    _p(f"\n📊 Categorical columns ({len(categorical_cols)}):")
    for c in categorical_cols:
        _p(f"   • {c}")

    # ── 8–10. Build scikit-learn preprocessing pipeline ─────────────
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,   numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    _p("\n✅ Preprocessing pipeline built:")
    _p(preprocessor)

    # ── 11. Train / test split (stratified) ─────────────────────────
    if forced_indices is not None:
        train_idx = forced_indices["train_idx"]
        test_idx = forced_indices["test_idx"]
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        train_customer_ids = customer_ids.iloc[train_idx].copy()
        test_customer_ids = customer_ids.iloc[test_idx].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        train_customer_ids = customer_ids.loc[X_train.index].copy()
        test_customer_ids = customer_ids.loc[X_test.index].copy()

    _p(f"\n✅ Train/test split (80/20, stratified):")
    _p(f"   X_train: {X_train.shape}   y_train: {y_train.shape}")
    _p(f"   X_test:  {X_test.shape}    y_test:  {y_test.shape}")
    _p(f"   Train churn rate: {y_train.mean():.4f}")
    _p(f"   Test  churn rate: {y_test.mean():.4f}")

    # ── 12. Fit on train only → transform both ──────────────────────
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    _p(f"\n✅ Preprocessing complete (no data leakage):")
    _p(f"   X_train_processed shape: {X_train_processed.shape}")
    _p(f"   X_test_processed  shape: {X_test_processed.shape}")
    _p(f"   Total engineered features: {len(feature_names)}")

    # ── Sanity checks ───────────────────────────────────────────────
    assert X_train_processed.shape[0] == y_train.shape[0]
    assert X_test_processed.shape[0]  == y_test.shape[0]
    assert X_train_processed.shape[1] == X_test_processed.shape[1]
    assert not np.isnan(X_train_processed).any(), \
        "NaNs remain in training set!"
    assert not np.isnan(X_test_processed).any(), \
        "NaNs remain in test set!"

    _p("\n✅ All sanity checks passed — data is ready for modelling.")

    _p(f"\n📋 Engineered feature names ({len(feature_names)} total):")
    for i, name in enumerate(feature_names, 1):
        _p(f"   {i:>3}. {name}")

    return {
        "X_train":            X_train,
        "X_test":             X_test,
        "y_train":            y_train,
        "y_test":             y_test,
        "X_train_processed":  X_train_processed,
        "X_test_processed":   X_test_processed,
        "preprocessor":       preprocessor,
        "feature_names":      feature_names,
        "numerical_cols":     numerical_cols,
        "categorical_cols":   categorical_cols,
        "X_train_raw":        X_train,
        "X_test_raw":         X_test,
        "train_customer_ids": train_customer_ids,
        "test_customer_ids":  test_customer_ids,
    }


# ─────────────────────────────────────────────────────────────────────
# Standalone execution (same output as the original script)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_and_preprocess(verbose=True)
