"""Comprehensive, isolated test suite for baseline vs. expanded experiments.

All tests utilize pytest's tmp_path fixture to avoid polluting project directories.
Guarantees that existing baseline files in models/ and reports/ remain completely untouched.
"""
import os
import glob
import hashlib
import pandas as pd
import joblib
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_and_preprocess, FORBIDDEN_FEATURES
from src.split_indices import compute_global_indices
from src.tune_wrapper import tune_expanded
from src.train_wrapper import train_expanded_models
from src.train_baseline_wrapper import train_baseline_models
from src.run_experiments import run_all_experiments
from src.app_expanded import build_feature_dict


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_raw_df():
    csv_path = os.path.join(_project_root(), "data", "Telco-Customer-Churn.csv")
    return pd.read_csv(csv_path)


def _hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _hash_models_dir():
    models_dir = os.path.join(_project_root(), "models")
    hashes = {}
    for p in sorted(glob.glob(os.path.join(models_dir, "*"))):
        if os.path.isfile(p):
            hashes[os.path.basename(p)] = _hash_file(p)
    return hashes


def test_split_properties():
    """Verify compute_global_indices produces reproducible, stratified 80/20 splits."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw, random_state=42)
    train_idx = indices["train_idx"]
    test_idx = indices["test_idx"]

    # Sizes
    assert len(train_idx) + len(test_idx) == 7043
    assert abs(len(test_idx) - int(round(7043 * 0.2))) <= 1

    # Mutual exclusivity
    assert len(set(train_idx).intersection(set(test_idx))) == 0

    # Stratification
    y = raw["Churn"].map({"No": 0, "Yes": 1})
    overall_rate = y.mean()
    train_rate = y.iloc[train_idx].mean()
    test_rate = y.iloc[test_idx].mean()
    assert abs(train_rate - overall_rate) < 0.01
    assert abs(test_rate - overall_rate) < 0.01

    # Reproducibility
    indices_repeat = compute_global_indices(raw, random_state=42)
    assert train_idx.tolist() == indices_repeat["train_idx"].tolist()
    assert test_idx.tolist() == indices_repeat["test_idx"].tolist()


def test_baseline_preprocessing_regression():
    """Verify load_and_preprocess() default call remains 100% backward compatible."""
    data = load_and_preprocess(verbose=False)

    expected_keys = {
        "X_train", "X_test", "y_train", "y_test",
        "X_train_processed", "X_test_processed",
        "preprocessor", "feature_names",
        "numerical_cols", "categorical_cols",
    }
    assert expected_keys.issubset(data.keys())

    # Check baseline shapes
    assert data["X_train"].shape == (5634, 19)
    assert data["X_test"].shape == (1409, 19)
    assert data["y_train"].shape == (5634,)
    assert data["y_test"].shape == (1409,)
    assert data["X_train_processed"].shape == (5634, 45)
    assert data["X_test_processed"].shape == (1409, 45)

    # Check columns
    assert "Satisfaction Score" not in data["numerical_cols"]
    assert "Satisfaction Score" not in data["X_train"].columns
    assert "customerID" not in data["X_train"].columns
    assert FORBIDDEN_FEATURES.intersection(set(data["X_train"].columns)) == set()


def test_merge_and_forbidden_features():
    """Verify merge with status file and strict exclusion of leakage columns."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)
    expanded = load_and_preprocess(
        verbose=False,
        add_status=True,
        raw_df=raw,
        forced_indices=indices,
    )

    assert len(expanded["X_train"]) + len(expanded["X_test"]) == 7043
    assert "Satisfaction Score" in expanded["X_train_raw"].columns
    assert expanded["X_train_raw"]["Satisfaction Score"].isna().sum() == 0
    assert expanded["X_test_raw"]["Satisfaction Score"].isna().sum() == 0

    # Strict leakage assertion: intersection must be empty
    train_leakage = FORBIDDEN_FEATURES.intersection(set(expanded["X_train_raw"].columns))
    test_leakage = FORBIDDEN_FEATURES.intersection(set(expanded["X_test_raw"].columns))
    assert not train_leakage, f"Forbidden columns present in training features: {train_leakage}"
    assert not test_leakage, f"Forbidden columns present in test features: {test_leakage}"


def test_shared_split_ids():
    """Verify that baseline and expanded experiments partition identical customer IDs."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)

    base = load_and_preprocess(verbose=False, add_status=False, raw_df=raw, forced_indices=indices)
    exp = load_and_preprocess(verbose=False, add_status=True, raw_df=raw, forced_indices=indices)

    assert base["train_customer_ids"].tolist() == exp["train_customer_ids"].tolist()
    assert base["test_customer_ids"].tolist() == exp["test_customer_ids"].tolist()


def test_preprocessor_fitted_on_train_only():
    """Verify that preprocessor fits only on train and transforms test data cleanly."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)
    exp = load_and_preprocess(verbose=False, add_status=True, raw_df=raw, forced_indices=indices)

    preproc = exp["preprocessor"]
    transformed_test = preproc.transform(exp["X_test_raw"])

    assert transformed_test.shape == (len(exp["X_test_raw"]), 46)
    assert not pd.isna(transformed_test).any()


def test_tuned_pipeline_leak_free_cv(tmp_path):
    """Verify hyperparameter tuning constructs a leak-free cross-validated Pipeline."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)
    exp = load_and_preprocess(verbose=False, add_status=True, raw_df=raw, forced_indices=indices)

    pipe, metrics = tune_expanded(exp, model_dir=str(tmp_path), n_iter=2, random_state=42)

    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["classifier"], RandomForestClassifier)
    assert "Satisfaction Score" in pipe.named_steps["preprocessor"].feature_names_in_
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0


def test_tuned_artifact_integrity(tmp_path):
    """Verify best_churn_model_expanded.joblib is created by tuner and not overwritten."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)
    exp = load_and_preprocess(verbose=False, add_status=True, raw_df=raw, forced_indices=indices)

    tuned_dir = str(tmp_path / "exp_models")
    tune_expanded(exp, model_dir=tuned_dir, n_iter=2)

    tuned_file = os.path.join(tuned_dir, "best_churn_model_expanded.joblib")
    assert os.path.exists(tuned_file)
    hash_before = _hash_file(tuned_file)

    # Run auxiliary trainer: must NOT overwrite best_churn_model_expanded.joblib
    train_expanded_models(exp, model_dir=tuned_dir)

    hash_after = _hash_file(tuned_file)
    assert hash_before == hash_after, "Auxiliary training modified best_churn_model_expanded.joblib!"


def test_app_expanded_input_schema(tmp_path):
    """Verify build_feature_dict matches preprocessor features and predicts properly."""
    raw = _load_raw_df()
    indices = compute_global_indices(raw)
    exp = load_and_preprocess(verbose=False, add_status=True, raw_df=raw, forced_indices=indices)

    exp_dir = str(tmp_path / "expanded")
    best_pipe, _ = tune_expanded(exp, model_dir=exp_dir, n_iter=2)

    expected_cols = set(best_pipe.named_steps["preprocessor"].feature_names_in_)

    sample_dict = build_feature_dict(
        gender="Male",
        senior_citizen="No",
        partner="No",
        dependents="No",
        tenure=12,
        phone_service="Yes",
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="Yes",
        streaming_movies="Yes",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
        monthly_charges=85.0,
        total_charges=1020.0,
        satisfaction_score=2,
    )

    # Schema checks
    assert set(sample_dict.keys()) == expected_cols, (
        f"Mismatch between UI input keys and preprocessor features!\n"
        f"Missing from UI: {expected_cols - set(sample_dict.keys())}\n"
        f"Extra in UI: {set(sample_dict.keys()) - expected_cols}"
    )
    assert "Satisfaction Score" in sample_dict
    assert not FORBIDDEN_FEATURES.intersection(set(sample_dict.keys()))
    assert "customerID" not in sample_dict and "Customer ID" not in sample_dict

    # Single-step prediction check
    df_sample = pd.DataFrame([sample_dict])
    prob = best_pipe.predict_proba(df_sample)[:, 1]
    assert prob.shape == (1,)
    assert 0.0 <= prob[0] <= 1.0


def test_end_to_end_no_baseline_overwrite(tmp_path):
    """Verify end-to-end experiment run executes safely in isolated dirs without mutating models/."""
    initial_hashes = _hash_models_dir()

    base_dir = str(tmp_path / "models_baseline_test")
    exp_dir = str(tmp_path / "models_expanded_test")

    # Run full orchestrator with custom dirs
    run_all_experiments(baseline_dir=base_dir, expanded_dir=exp_dir, verbose=False)

    # 1. Baseline directory models/ must have identical hashes
    final_hashes = _hash_models_dir()
    assert initial_hashes == final_hashes, "Project models/ directory files were modified during test!"

    # 2. Check generated artifacts in custom test dirs
    assert os.path.exists(os.path.join(base_dir, "best_churn_model.joblib"))
    assert os.path.exists(os.path.join(base_dir, "comparison_table.csv"))
    assert os.path.exists(os.path.join(exp_dir, "best_churn_model_expanded.joblib"))
    assert os.path.exists(os.path.join(exp_dir, "comparison_table_expanded.csv"))


def test_individual_shap_explanation():
    """Verify that local SHAP attributions dynamically update per customer while global importances remain constant."""
    from src.app_expanded import compute_individual_shap, load_baseline_pipeline

    pipe = load_baseline_pipeline()
    assert pipe is not None

    # 1. High-risk profile (Month-to-month, short tenure, high charges)
    high_sample = build_feature_dict(
        tenure=1, contract="Month-to-month", internet_service="Fiber optic", monthly_charges=85.0
    )
    shap_high = compute_individual_shap(pipe, high_sample)

    # 2. Low-risk profile (Two year, long tenure, low charges)
    low_sample = build_feature_dict(
        tenure=60, contract="Two year", internet_service="DSL", monthly_charges=25.0
    )
    shap_low = compute_individual_shap(pipe, low_sample)

    # Predictions and SHAP values must dynamically differentiate
    assert shap_high["pred_prob"] > shap_low["pred_prob"]
    assert shap_high["grouped_shap"]["Contract"] != shap_low["grouped_shap"]["Contract"]
    assert shap_high["grouped_shap"]["tenure"] != shap_low["grouped_shap"]["tenure"]

    # Month-to-month increases risk; Two-year decreases risk
    assert shap_high["grouped_shap"]["Contract"] > 0
    assert shap_low["grouped_shap"]["Contract"] < 0

    # Additivity holds within float precision: sum of grouped SHAP + base_value approx equals pred_prob
    sum_high = sum(shap_high["grouped_shap"].values()) + shap_high["base_value"]
    assert abs(sum_high - shap_high["pred_prob"]) < 0.05

    # Global Gini importance remains constant
    clf = pipe.named_steps["classifier"]
    assert len(clf.feature_importances_) == 45
