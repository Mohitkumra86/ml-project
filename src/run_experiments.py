"""End-to-end experiment runner for Baseline vs. Expanded Telco Churn experiments.

Loads the raw CSV once, creates shared stratified train/test indices,
and trains both the baseline experiment (in models_baseline_experiment/)
and the expanded experiment (in models_expanded/).
Guarantees zero mutation of project-root models/.

Usage:
    python -m src.run_experiments
    python -m src.run_experiments --baseline_dir <path> --expanded_dir <path>
"""
import os
import argparse
import pandas as pd

from preprocess import load_and_preprocess
from src.split_indices import compute_global_indices
from src.train_baseline_wrapper import train_baseline_models
from src.tune_wrapper import tune_expanded
from src.train_wrapper import train_expanded_models


def run_all_experiments(
    baseline_dir: str = "models_baseline_experiment",
    expanded_dir: str = "models_expanded",
    verbose: bool = True,
):
    """Run baseline and expanded experiments with shared stratified split."""
    _p = print if verbose else (lambda *a, **k: None)

    _p("=" * 72)
    _p("  TELCO CHURN EXPERIMENT COMPARISON: BASELINE vs. EXPANDED DATASET")
    _p("=" * 72)

    # 1. Load raw CSV once
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv_path = os.path.join(project_root, "data", "Telco-Customer-Churn.csv")
    raw_df = pd.read_csv(raw_csv_path)
    _p(f"✅ Loaded raw CSV once: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    # 2. Compute shared stratified train/test indices
    indices = compute_global_indices(raw_df, random_state=42)
    _p(f"✅ Generated shared stratified splits: {len(indices['train_idx'])} train / {len(indices['test_idx'])} test")

    # 3. Baseline Pipeline
    _p("\n" + "─" * 40)
    _p(f"▶ 1/2 Training Baseline Experiment (output: {baseline_dir})")
    _p("─" * 40)
    baseline_data = load_and_preprocess(
        verbose=False,
        add_status=False,
        raw_df=raw_df,
        forced_indices=indices,
    )
    df_base_results = train_baseline_models(baseline_data, model_dir=baseline_dir)
    _p(f"✅ Baseline training complete. Models saved to {baseline_dir}/")

    # 4. Expanded Pipeline
    _p("\n" + "─" * 40)
    _p(f"▶ 2/2 Training & Tuning Expanded Experiment (output: {expanded_dir})")
    _p("─" * 40)
    expanded_data = load_and_preprocess(
        verbose=False,
        add_status=True,
        raw_df=raw_df,
        forced_indices=indices,
    )

    # 4a. Leak-free hyperparameter tuning for expanded model
    _p("   • Running leak-free CV tuning on raw training data...")
    best_pipe, tuned_metrics = tune_expanded(expanded_data, model_dir=expanded_dir)
    _p(f"   • Tuned Expanded Model: Test F1={tuned_metrics['F1 Score']:.4f}, ROC-AUC={tuned_metrics['ROC-AUC']:.4f}")
    _p(f"   • Saved to {expanded_dir}/best_churn_model_expanded.joblib")

    # 4b. Auxiliary expanded models (never writes best_churn_model_expanded.joblib)
    _p("   • Training auxiliary classifiers on expanded features...")
    df_exp_results = train_expanded_models(expanded_data, model_dir=expanded_dir)
    _p(f"✅ Expanded models saved to {expanded_dir}/")

    # 5. Summary comparison
    _p("\n" + "=" * 72)
    _p("  SUMMARY EXPERIMENT RESULTS")
    _p("=" * 72)
    _p("\nBaseline Experiment Results:")
    _p(df_base_results.to_string(index=False))
    _p("\nExpanded Experiment Tuned Random Forest:")
    summary_row = pd.DataFrame([{
        "Model": "Tuned Random Forest (Expanded)",
        "Accuracy": tuned_metrics["Accuracy"],
        "Precision": tuned_metrics["Precision"],
        "Recall": tuned_metrics["Recall"],
        "F1 Score": tuned_metrics["F1 Score"],
        "ROC-AUC": tuned_metrics["ROC-AUC"],
    }])
    _p(summary_row.to_string(index=False))
    _p("\n⚠️ Note: The expanded model includes 'Satisfaction Score' which may exhibit")
    _p("   target leakage due to retrospective measurement. The baseline model")
    _p("   remains the production reference standard.")
    _p("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run Baseline and Expanded Telco ML Experiments")
    parser.add_argument(
        "--baseline_dir",
        default="models_baseline_experiment",
        help="Directory to save baseline experiment outputs (default: models_baseline_experiment)",
    )
    parser.add_argument(
        "--expanded_dir",
        default="models_expanded",
        help="Directory to save expanded experiment outputs (default: models_expanded)",
    )
    args = parser.parse_args()
    run_all_experiments(baseline_dir=args.baseline_dir, expanded_dir=args.expanded_dir)


if __name__ == "__main__":
    main()
