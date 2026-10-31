"""
Telecom Customer Churn – Prediction Script
============================================
Loads the saved end-to-end pipeline (preprocessor + classifier) and
predicts churn probability, prediction, and risk level for new
customer data.

Usage
-----
    python predict.py                  # runs built-in sample customers
    from predict import predict_churn  # import for use in other scripts
"""

import os
import pickle
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Feature schema — must match the training data exactly
# ─────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "gender",            # Female / Male
    "SeniorCitizen",     # 0 / 1
    "Partner",           # Yes / No
    "Dependents",        # Yes / No
    "tenure",            # integer (months)
    "PhoneService",      # Yes / No
    "MultipleLines",     # Yes / No / No phone service
    "InternetService",   # DSL / Fiber optic / No
    "OnlineSecurity",    # Yes / No / No internet service
    "OnlineBackup",      # Yes / No / No internet service
    "DeviceProtection",  # Yes / No / No internet service
    "TechSupport",       # Yes / No / No internet service
    "StreamingTV",       # Yes / No / No internet service
    "StreamingMovies",   # Yes / No / No internet service
    "Contract",          # Month-to-month / One year / Two year
    "PaperlessBilling",  # Yes / No
    "PaymentMethod",     # Electronic check / Mailed check /
                         # Bank transfer (automatic) /
                         # Credit card (automatic)
    "MonthlyCharges",    # float
    "TotalCharges",      # float
]

# ─────────────────────────────────────────────────────────────────────
# Load the saved pipeline
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "models", "best_churn_model.pkl")


def _load_pipeline():
    """Load and cache the saved pipeline."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_pipeline = None  # lazy-loaded singleton


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def _risk_level(probability: float) -> str:
    """Map churn probability to a human-readable risk level."""
    if probability >= 0.75:
        return "🔴 HIGH"
    elif probability >= 0.45:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"


# ─────────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────────
def predict_churn(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Keys must match FEATURE_COLUMNS.  Example::

            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                ...
                "MonthlyCharges": 70.35,
                "TotalCharges": 1397.5,
            }

    Returns
    -------
    dict with keys:
        probability  – float in [0, 1]
        prediction   – 0 or 1
        label        – "Will Churn" / "Will Not Churn"
        risk_level   – "🔴 HIGH" / "🟡 MEDIUM" / "🟢 LOW"
    """
    pipeline = _get_pipeline()

    # Build a single-row DataFrame in the exact column order
    df = pd.DataFrame([customer], columns=FEATURE_COLUMNS)

    # Ensure correct dtypes (match training data)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    prob = pipeline.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "probability": prob,
        "prediction":  pred,
        "label":       "Will Churn" if pred == 1 else "Will Not Churn",
        "risk_level":  _risk_level(prob),
    }


def _print_result(name: str, customer: dict):
    """Pretty-print a prediction."""
    result = predict_churn(customer)
    print(f"\n{'─' * 55}")
    print(f"  👤 {name}")
    print(f"{'─' * 55}")

    # Show key customer details
    print(f"   Contract:       {customer['Contract']}")
    print(f"   Tenure:         {customer['tenure']} months")
    print(f"   Monthly:        ${customer['MonthlyCharges']}")
    print(f"   Internet:       {customer['InternetService']}")
    print(f"   Payment:        {customer['PaymentMethod']}")

    print()
    print(f"   Churn Probability:  {result['probability'] * 100:.1f}%")
    print(f"   Prediction:         {result['label']}")
    print(f"   Risk Level:         {result['risk_level']}")


# ─────────────────────────────────────────────────────────────────────
# Sample customers for testing
# ─────────────────────────────────────────────────────────────────────
SAMPLE_CUSTOMERS = {

    "High-Risk Customer (Alice)": {
        "gender":           "Female",
        "SeniorCitizen":    1,
        "Partner":          "No",
        "Dependents":       "No",
        "tenure":           2,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "No",
        "StreamingMovies":  "No",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   85.50,
        "TotalCharges":     171.00,
    },

    "Medium-Risk Customer (Bob)": {
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "No",
        "tenure":           15,
        "PhoneService":     "Yes",
        "MultipleLines":    "Yes",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "Yes",
        "DeviceProtection": "Yes",
        "TechSupport":      "No",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "No",
        "Contract":         "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Credit card (automatic)",
        "MonthlyCharges":   75.20,
        "TotalCharges":     1128.00,
    },

    "Low-Risk Customer (Carol)": {
        "gender":           "Female",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "Yes",
        "tenure":           60,
        "PhoneService":     "Yes",
        "MultipleLines":    "Yes",
        "InternetService":  "DSL",
        "OnlineSecurity":   "Yes",
        "OnlineBackup":     "Yes",
        "DeviceProtection": "Yes",
        "TechSupport":      "Yes",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "Yes",
        "Contract":         "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod":    "Bank transfer (automatic)",
        "MonthlyCharges":   85.10,
        "TotalCharges":     5106.00,
    },

    "Edge Case – No Internet (Dave)": {
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "No",
        "Dependents":       "Yes",
        "tenure":           40,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "No",
        "OnlineSecurity":   "No internet service",
        "OnlineBackup":     "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport":      "No internet service",
        "StreamingTV":      "No internet service",
        "StreamingMovies":  "No internet service",
        "Contract":         "One year",
        "PaperlessBilling": "No",
        "PaymentMethod":    "Mailed check",
        "MonthlyCharges":   20.05,
        "TotalCharges":     802.00,
    },
}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TELECOM CHURN – PREDICTION ENGINE")
    print("=" * 55)
    print(f"  Model: {MODEL_PATH}")

    for name, cust in SAMPLE_CUSTOMERS.items():
        _print_result(name, cust)

    print(f"\n{'=' * 55}")
    print("  ✅ All predictions completed successfully")
    print(f"{'=' * 55}\n")
