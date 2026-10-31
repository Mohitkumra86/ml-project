"""Telecom Customer Churn Intelligence – Midnight Observatory Dashboard
====================================================================
Production analytics dashboard featuring the tuned baseline Random Forest
as the primary predictor, individual SHAP feature attributions, and an
isolated experimental telemetry lab with retrospective leakage disclosure.
"""

import os
import pickle
import altair as alt
import joblib
import pandas as pd
import shap
import streamlit as st


def build_feature_dict(
    gender: str = "Female",
    senior_citizen: str = "No",
    partner: str = "Yes",
    dependents: str = "No",
    tenure: int = 1,
    phone_service: str = "No",
    multiple_lines: str = "No phone service",
    internet_service: str = "DSL",
    online_security: str = "No",
    online_backup: str = "Yes",
    device_protection: str = "No",
    tech_support: str = "No",
    streaming_tv: str = "No",
    streaming_movies: str = "No",
    contract: str = "Month-to-month",
    paperless_billing: str = "Yes",
    payment_method: str = "Electronic check",
    monthly_charges: float = 29.85,
    total_charges: float = 29.85,
    satisfaction_score: int = 3,
) -> dict:
    """Build the raw prediction input dictionary.

    Retains the full 20-feature signature required by the test suite
    and experimental model pipeline.
    """
    return {
        "gender": str(gender),
        "SeniorCitizen": 1 if str(senior_citizen).strip() in ("Yes", "1", "True", 1, True) else 0,
        "Partner": str(partner),
        "Dependents": str(dependents),
        "tenure": int(tenure),
        "PhoneService": str(phone_service),
        "MultipleLines": str(multiple_lines),
        "InternetService": str(internet_service),
        "OnlineSecurity": str(online_security),
        "OnlineBackup": str(online_backup),
        "DeviceProtection": str(device_protection),
        "TechSupport": str(tech_support),
        "StreamingTV": str(streaming_tv),
        "StreamingMovies": str(streaming_movies),
        "Contract": str(contract),
        "PaperlessBilling": str(paperless_billing),
        "PaymentMethod": str(payment_method),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "Satisfaction Score": int(satisfaction_score),
    }


def extract_baseline_features(features: dict) -> dict:
    """Extract the 19 raw features expected by the production baseline model."""
    return {k: v for k, v in features.items() if k != "Satisfaction Score"}


def get_baseline_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "best_churn_model.pkl")


def get_expanded_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models_expanded", "best_churn_model_expanded.joblib")


@st.cache_resource
def load_baseline_pipeline():
    path = get_baseline_model_path()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_expanded_pipeline():
    path = get_expanded_model_path()
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def get_baseline_tree_explainer(_clf):
    """Cache the SHAP TreeExplainer instance for high-performance inference."""
    return shap.TreeExplainer(_clf)


def compute_individual_shap(baseline_pipe, customer_features: dict) -> dict:
    """Compute local SHAP feature attributions for a single customer profile.

    Maps 45 one-hot encoded tree contributions back to the 19 human-readable
    customer attributes, distinguishing positive risk drivers from protective factors.
    """
    preproc = baseline_pipe.named_steps["preprocessor"]
    clf = baseline_pipe.named_steps["classifier"]

    base_dict = extract_baseline_features(customer_features)
    df_input = pd.DataFrame([base_dict])
    X_trans = preproc.transform(df_input)

    explainer = get_baseline_tree_explainer(clf)
    shap_vals = explainer(X_trans)

    # Values for Class 1 (Churn)
    vals = shap_vals.values[0, :, 1] if len(shap_vals.values.shape) == 3 else shap_vals.values[0]
    base_val_raw = shap_vals.base_values[0, 1] if len(shap_vals.base_values.shape) == 2 else shap_vals.base_values[0]

    base_value = float(base_val_raw) * 100.0
    pred_prob = float(clf.predict_proba(X_trans)[0, 1]) * 100.0

    feature_names = preproc.get_feature_names_out()
    raw_cols = df_input.columns.tolist()

    grouped_shap = {}
    for feat, val in zip(feature_names, vals):
        for rc in raw_cols:
            if feat.startswith(f"num__{rc}") or feat.startswith(f"cat__{rc}_"):
                grouped_shap[rc] = grouped_shap.get(rc, 0.0) + (float(val) * 100.0)
                break

    drivers = []
    protective = []
    for col, net_pct in grouped_shap.items():
        val_display = str(base_dict.get(col, ""))
        if net_pct > 0.05:
            drivers.append((col, val_display, net_pct))
        elif net_pct < -0.05:
            protective.append((col, val_display, net_pct))

    drivers.sort(key=lambda x: x[2], reverse=True)
    protective.sort(key=lambda x: x[2])  # Most negative first

    return {
        "base_value": base_value,
        "pred_prob": pred_prob,
        "risk_drivers": drivers,
        "protective_factors": protective,
        "grouped_shap": grouped_shap,
    }


def build_shap_chart(drivers: list, protective: list, top_n: int = 5):
    """Build a horizontal Altair attribution chart separating risk drivers from protective factors."""
    items = []
    for feat, val, pct in drivers[:top_n]:
        items.append({
            "Attribute": f"{feat}: {val}",
            "Impact": round(pct, 2),
            "Type": "Increases Risk (+)",
        })
    for feat, val, pct in protective[:top_n]:
        items.append({
            "Attribute": f"{feat}: {val}",
            "Impact": round(pct, 2),
            "Type": "Decreases Risk (-)",
        })

    if not items:
        return None

    df_plot = pd.DataFrame(items)
    chart = (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusEnd=3, height=20)
        .encode(
            y=alt.Y(
                "Attribute:N",
                sort=alt.SortField("Impact", order="descending"),
                title=None,
                axis=alt.Axis(
                    labelColor="#CBD5E1",
                    labelFontSize=12,
                    labelFontWeight=500,
                    labelLimit=240,
                    tickSize=0,
                ),
            ),
            x=alt.X(
                "Impact:Q",
                title="Contribution to Churn Probability (% points)",
                axis=alt.Axis(
                    labelColor="#94A3B8",
                    titleColor="#A3AEC2",
                    titleFontSize=11,
                    gridColor="#1E293B",
                    gridOpacity=0.6,
                    labelExpr="datum.value > 0 ? '+' + datum.value + '%' : datum.value + '%'",
                ),
            ),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Increases Risk (+)", "Decreases Risk (-)"],
                    range=["#E06D75", "#69D6BE"],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    labelColor="#CBD5E1",
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip("Attribute:N", title="Customer Attribute"),
                alt.Tooltip("Impact:Q", title="Probability Shift (% pts)", format="+.2f"),
                alt.Tooltip("Type:N", title="Effect"),
            ],
        )
        .properties(
            height=max(160, 26 * len(df_plot) + 40),
            padding={"top": 5, "left": 10, "right": 20, "bottom": 10},
        )
        .configure_view(stroke=None)
        .configure_axis(domainColor="#29344A")
    )
    return chart


def build_global_importance_chart(baseline_pipe, top_n: int = 7):
    """Build a compact horizontal Altair bar chart for global Gini feature importance."""
    try:
        clf = baseline_pipe.named_steps["classifier"]
        prep = baseline_pipe.named_steps["preprocessor"]
        raw_names = prep.get_feature_names_out()
        importances = clf.feature_importances_

        cleaned_names = []
        for n in raw_names:
            clean = n.replace("num__", "").replace("cat__", "")
            if "_" in clean:
                parts = clean.split("_", 1)
                clean = f"{parts[0]} = {parts[1]}"
            cleaned_names.append(clean)

        df_imp = (
            pd.DataFrame({"Feature": cleaned_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n)
            .copy()
        )
        df_imp["Percentage"] = (df_imp["Importance"] * 100.0).round(1)

        chart = (
            alt.Chart(df_imp)
            .mark_bar(cornerRadiusEnd=3, color="#A69BFF", height=18)
            .encode(
                y=alt.Y(
                    "Feature:N",
                    sort=alt.SortField("Importance", order="descending"),
                    title=None,
                    axis=alt.Axis(
                        labelColor="#CBD5E1",
                        labelFontSize=12,
                        labelFontWeight=500,
                        labelLimit=230,
                        tickSize=0,
                    ),
                ),
                x=alt.X(
                    "Percentage:Q",
                    title="Gini Impurity Importance (% of Total)",
                    axis=alt.Axis(
                        labelColor="#94A3B8",
                        titleColor="#A3AEC2",
                        titleFontSize=11,
                        gridColor="#1E293B",
                        gridOpacity=0.6,
                        labelExpr="datum.value + '%'",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Feature:N", title="Feature"),
                    alt.Tooltip("Percentage:Q", title="Importance (%)", format=".1f"),
                ],
            )
            .properties(
                height=max(160, 26 * len(df_imp) + 30),
                padding={"top": 5, "left": 10, "right": 20, "bottom": 10},
            )
            .configure_view(stroke=None)
            .configure_axis(domainColor="#29344A")
        )
        return chart, df_imp
    except Exception:
        return None, None


def main():
    st.set_page_config(
        page_title="Customer Churn Observatory",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Scoped "Midnight Observatory" Design System
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Root Reset & Theme */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F1F3FA;
        }

        .stApp {
            background-color: #0B1020;
            color: #F1F3FA;
        }

        /* Top Brand Header */
        .obs-header-container {
            background: #121A2B;
            border: 1px solid #253248;
            border-radius: 8px;
            padding: 1.35rem 1.75rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .obs-brand-eyebrow {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #A69BFF;
            margin-bottom: 0.25rem;
        }

        .obs-brand-title {
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: #F1F3FA;
            margin: 0;
            line-height: 1.2;
        }

        .obs-brand-subtitle {
            font-size: 0.86rem;
            color: #A3AEC2;
            margin: 0.3rem 0 0 0;
            max-width: 680px;
            line-height: 1.45;
        }

        .obs-pill-cluster {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .obs-pill {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 4px;
            border: 1px solid #253248;
            background: #182236;
            color: #A3AEC2;
            letter-spacing: 0.01em;
        }

        .obs-pill-accent {
            border-color: rgba(166, 155, 255, 0.4);
            background: rgba(166, 155, 255, 0.12);
            color: #A69BFF;
        }

        .obs-pill-teal {
            border-color: rgba(105, 214, 190, 0.4);
            background: rgba(105, 214, 190, 0.12);
            color: #69D6BE;
        }

        /* Tabs Navigation */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: #A3AEC2 !important;
            font-weight: 600 !important;
            font-size: 0.925rem !important;
            padding: 0.7rem 1.5rem !important;
            border: none !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #A69BFF !important;
            border-bottom: 2px solid #A69BFF !important;
        }

        /* Section Cards */
        .obs-section-card {
            background: #121A2B;
            border: 1px solid #253248;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .obs-section-header {
            border-bottom: 1px solid #253248;
            padding-bottom: 0.65rem;
            margin-bottom: 1.15rem;
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .obs-section-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #F1F3FA;
            margin: 0;
            letter-spacing: -0.01em;
        }

        .obs-section-caption {
            font-size: 0.775rem;
            color: #A3AEC2;
            margin: 0;
        }

        /* Probability Result Display */
        .obs-result-panel {
            background: #121A2B;
            border: 1px solid #253248;
            border-radius: 8px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }

        .obs-stat-figure {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.03em;
            margin: 0.4rem 0;
        }

        .obs-badge-risk {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.3rem 0.75rem;
            border-radius: 4px;
            display: inline-block;
        }

        .risk-low-badge {
            background: rgba(105, 214, 190, 0.12);
            border: 1px solid rgba(105, 214, 190, 0.35);
            color: #69D6BE;
        }

        .risk-med-badge {
            background: rgba(240, 201, 135, 0.12);
            border: 1px solid rgba(240, 201, 135, 0.35);
            color: #F0C987;
        }

        .risk-high-badge {
            background: rgba(224, 109, 117, 0.12);
            border: 1px solid rgba(224, 109, 117, 0.35);
            color: #E06D75;
        }

        /* SHAP Individual Contribution Row */
        .obs-shap-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            background: #182236;
            border: 1px solid #253248;
            border-radius: 5px;
            margin-bottom: 0.4rem;
            font-size: 0.82rem;
        }

        .obs-shap-feat {
            font-weight: 600;
            color: #F1F3FA;
        }

        .obs-shap-val {
            color: #A3AEC2;
            font-size: 0.775rem;
            margin-left: 0.4rem;
        }

        .obs-shap-delta {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.82rem;
        }

        /* Primary Button */
        div.stButton > button:first-child {
            background-color: #A69BFF !important;
            color: #0B1020 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.7rem 1.5rem !important;
            font-size: 0.925rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(166, 155, 255, 0.18) !important;
        }

        div.stButton > button:first-child:hover {
            background-color: #B9B0FF !important;
            box-shadow: 0 6px 16px rgba(166, 155, 255, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        /* Form Inputs */
        div[data-baseweb="select"] > div {
            background-color: #182236 !important;
            border-color: #253248 !important;
            color: #F1F3FA !important;
            border-radius: 6px !important;
        }

        div[data-baseweb="input"] > div {
            background-color: #182236 !important;
            border-color: #253248 !important;
            color: #F1F3FA !important;
            border-radius: 6px !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #A3AEC2 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }

        .obs-footnote {
            font-size: 0.76rem;
            color: #717D96;
            line-height: 1.5;
            margin-top: 0.65rem;
        }

        /* Sensitivity Row */
        .obs-sens-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.84rem;
        }

        .obs-sens-label {
            width: 140px;
            color: #A3AEC2;
            font-weight: 500;
        }

        .obs-sens-track {
            flex-grow: 1;
            height: 10px;
            background: #182236;
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid #253248;
        }

        .obs-sens-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .obs-sens-value {
            width: 55px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #F1F3FA;
        }
    </style>
    """, unsafe_allow_html=True)

    # Top Brand Header
    st.markdown("""
    <div class="obs-header-container">
        <div>
            <div class="obs-brand-eyebrow">Telecom Retention Intelligence</div>
            <h1 class="obs-brand-title">Customer Churn Observatory</h1>
            <p class="obs-brand-subtitle">
                Operational risk assessment powered by the tuned Random Forest baseline model,
                featuring local SHAP attribution and global model transparency.
            </p>
        </div>
        <div class="obs-pill-cluster">
            <span class="obs-pill">Cohort: 7,043 Accounts (IBM Telco)</span>
            <span class="obs-pill obs-pill-accent">Primary Model: Tuned Random Forest</span>
            <span class="obs-pill obs-pill-teal">Test ROC-AUC: 0.840</span>
            <span class="obs-pill" style="border-color:#F0C987; color:#F0C987;">Decision Threshold: 50%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load pipelines
    baseline_pipe = load_baseline_pipeline()
    expanded_pipe = load_expanded_pipeline()

    if baseline_pipe is None:
        st.error(f"Baseline model artifact not found at `{get_baseline_model_path()}`.")
        return

    # Two Main Tabs: Production Workflow & Experimental Lab
    tab_production, tab_experimental = st.tabs([
        "⚡ Primary Risk Assessment (Baseline Model)",
        "🔬 Experimental Telemetry Lab (Status & Satisfaction)",
    ])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1: PRODUCTION RISK ASSESSMENT (BASELINE MODEL)
    # ─────────────────────────────────────────────────────────────────
    with tab_production:
        st.markdown('<div class="obs-section-card"><div class="obs-section-header">'
                    '<div>'
                    '<p class="obs-section-title">Customer Account Profile (19 Standard Operational Features)</p>'
                    '<p class="obs-section-caption">Demographic, subscription, and contractual parameters expected by the baseline model</p>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)

        def _sync_prod_total():
            t = st.session_state.get("prod_tenure", 12)
            m = st.session_state.get("prod_monthly", 75.50)
            calc = float(round(t * m, 2)) if t > 0 else float(m)
            st.session_state["prod_total"] = min(max(calc, 0.0), 9000.0)

        if "prod_total" not in st.session_state:
            init_t = st.session_state.get("prod_tenure", 12)
            init_m = st.session_state.get("prod_monthly", 75.50)
            st.session_state["prod_total"] = min(max(float(round(init_t * init_m, 2)) if init_t > 0 else float(init_m), 0.0), 9000.0)

        col_demog, col_serv, col_bill = st.columns(3)

        with col_demog:
            st.markdown("<p style='font-size:0.835rem; font-weight:700; color:#CBD5E1; margin-bottom:0.4rem;'>👤 Customer Profile</p>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0, key="prod_gender")
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0, key="prod_senior")
            partner = st.selectbox("Partner", ["Yes", "No"], index=0, key="prod_partner")
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0, key="prod_dependents")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1, key="prod_tenure", on_change=_sync_prod_total)

        with col_serv:
            st.markdown("<p style='font-size:0.835rem; font-weight:700; color:#CBD5E1; margin-bottom:0.4rem;'>📡 Subscribed Services</p>", unsafe_allow_html=True)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0, key="prod_phone")
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0, key="prod_multilines")
            internet_service = st.selectbox("Internet Service Tier", ["DSL", "Fiber optic", "No"], index=1, key="prod_internet")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0, key="prod_sec")
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=1, key="prod_bak")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0, key="prod_dev")
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0, key="prod_sup")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0, key="prod_tv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=0, key="prod_mov")

        with col_bill:
            st.markdown("<p style='font-size:0.835rem; font-weight:700; color:#CBD5E1; margin-bottom:0.4rem;'>💳 Billing & Contract</p>", unsafe_allow_html=True)
            contract = st.selectbox("Contractual Term", ["Month-to-month", "One year", "Two year"], index=0, key="prod_contract")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0, key="prod_paperless")
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
                key="prod_payment",
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=75.50, step=0.50, key="prod_monthly", on_change=_sync_prod_total)
            calc_total = float(round(tenure * monthly_charges, 2)) if tenure > 0 else float(monthly_charges)
            total_charges = st.number_input("Cumulative Total Charges ($)", min_value=0.0, max_value=9000.0, step=10.0, key="prod_total")
            if tenure > 3 and abs(total_charges - calc_total) > (monthly_charges * 3):
                st.caption(f"ℹ️ Consistency Note: At {tenure} months and ${monthly_charges:.2f}/mo, typical charges are ~${calc_total:.2f}. User value ${total_charges:.2f} is retained.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Primary Predict Action Button
        st.button("Assess Customer Churn Risk", type="primary", use_container_width=True)

        # Build feature dictionary using standard 19 baseline attributes
        full_feature_dict = build_feature_dict(
            gender=gender,
            senior_citizen=senior,
            partner=partner,
            dependents=dependents,
            tenure=tenure,
            phone_service=phone_service,
            multiple_lines=multiple_lines,
            internet_service=internet_service,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            contract=contract,
            paperless_billing=paperless,
            payment_method=payment,
            monthly_charges=monthly_charges,
            total_charges=total_charges,
            satisfaction_score=3,
        )

        # Compute SHAP local attributions for this individual customer profile
        shap_res = compute_individual_shap(baseline_pipe, full_feature_dict)
        pct_base = shap_res["pred_prob"]
        base_val = shap_res["base_value"]

        # Risk Classification & Decision Boundary
        binary_pred = "Likely to Churn" if pct_base >= 50.0 else "Likely to Retain"
        if pct_base < 45.0:
            risk_class = "Low Attrition Risk"
            badge_style = "risk-low-badge"
            gauge_color = "#69D6BE"
            risk_desc = (
                "Account reflects solid contractual retention indicators and below-average churn propensity. "
                "Standard service delivery and SLA monitoring advised."
            )
        elif pct_base < 65.0:
            risk_class = "Moderate Attrition Risk"
            badge_style = "risk-med-badge"
            gauge_color = "#F0C987"
            risk_desc = (
                "Account displays mixed retention signals near the decision threshold. "
                "Proactive customer success engagement recommended."
            )
        else:
            risk_class = "Elevated Attrition Risk"
            badge_style = "risk-high-badge"
            gauge_color = "#E06D75"
            risk_desc = (
                "Account exhibits multiple structural churn risk factors. "
                "Targeted outreach and contract term extension incentives advised."
            )

        # Production Results Display Panel
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="obs-result-panel">', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.1, 1.9])

        with res_col1:
            st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#A3AEC2; margin:0; font-weight:700;'>Predicted Churn Probability</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='obs-stat-figure' style='color:{gauge_color};'>{pct_base:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='obs-badge-risk {badge_style}'>{risk_class}</span>", unsafe_allow_html=True)

            # Compact Gauge with 50% Threshold Marker
            st.markdown(f"""
            <div style="margin-top:1.15rem;">
                <div style="height:8px; background:#182236; border-radius:4px; overflow:hidden; border:1px solid #253248; position:relative;">
                    <div style="width:{min(max(pct_base, 2.0), 100.0):.1f}%; height:100%; background:{gauge_color}; border-radius:4px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#717D96; margin-top:0.35rem; font-family:'JetBrains Mono', monospace;">
                    <span>0% (Low)</span>
                    <span style="color:#A3AEC2;">Threshold: 50.0%</span>
                    <span>100% (High)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#A3AEC2; margin:0; font-weight:700;'>Decision & Contextual Summary</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='color:#CBD5E1; font-size:0.875rem; margin:0.4rem 0 0.85rem 0; line-height:1.45;'>"
                f"Binary Classification: <strong style='color:#F1F3FA;'>{binary_pred}</strong> (evaluated at the 50.0% decision threshold). "
                f"{risk_desc} <em>Note: Probability estimates reflect statistical propensity based on training patterns rather than certainty.</em>"
                f"</p>",
                unsafe_allow_html=True,
            )

            st.markdown(f"""
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:0.5rem; font-size:0.8rem;">
                <div style="background:#182236; padding:0.55rem 0.8rem; border-radius:5px; border:1px solid #253248;">
                    <span style="color:#A3AEC2;">Contract:</span> <strong style="color:#F1F3FA;">{contract}</strong>
                </div>
                <div style="background:#182236; padding:0.55rem 0.8rem; border-radius:5px; border:1px solid #253248;">
                    <span style="color:#A3AEC2;">Tenure:</span> <strong style="color:#F1F3FA;">{tenure} months</strong>
                </div>
                <div style="background:#182236; padding:0.55rem 0.8rem; border-radius:5px; border:1px solid #253248;">
                    <span style="color:#A3AEC2;">Payment:</span> <strong style="color:#F1F3FA;">{payment}</strong>
                </div>
                <div style="background:#182236; padding:0.55rem 0.8rem; border-radius:5px; border:1px solid #253248;">
                    <span style="color:#A3AEC2;">Monthly Charges:</span> <strong style="color:#F1F3FA;">${monthly_charges:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────
        # INDIVIDUAL-LEVEL EXPLANATION: "Why This Customer Has This Risk"
        # ─────────────────────────────────────────────────────────────
        st.markdown('<div class="obs-section-card"><div class="obs-section-header">'
                    '<div>'
                    '<p class="obs-section-title">Why This Customer Has This Risk</p>'
                    '<p class="obs-section-caption">Individual SHAP (SHapley Additive exPlanations) attribution quantifying feature impact for this active customer</p>'
                    '</div>'
                    '<div style="font-size:0.75rem; color:#A3AEC2; background:#182236; padding:0.3rem 0.65rem; border-radius:4px; border:1px solid #253248;">'
                    f'Ensemble Base Prior: <strong style="color:#F1F3FA;">{base_val:.1f}%</strong>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)

        net_shift = pct_base - base_val
        shift_sign = "+" if net_shift >= 0 else ""
        shift_color = "#E06D75" if net_shift > 0 else "#69D6BE"

        st.markdown(f"""
        <p style="font-size:0.835rem; color:#CBD5E1; margin-bottom:0.85rem; line-height:1.5;">
            The model starts from the cohort average baseline prior of <strong>{base_val:.1f}%</strong>
            (reflecting the 3:1 recall penalty applied during training).
            This customer's operational parameters produce a net shift of
            <strong style="color:{shift_color};">{shift_sign}{net_shift:.1f}% points</strong>
            to arrive at the predicted <strong>{pct_base:.1f}%</strong> churn likelihood.
        </p>
        """, unsafe_allow_html=True)

        # Render Horizontal Contribution Chart
        shap_chart = build_shap_chart(shap_res["risk_drivers"], shap_res["protective_factors"], top_n=5)
        if shap_chart is not None:
            st.altair_chart(shap_chart, use_container_width=True)
        else:
            st.info("Feature attributions fall within standard baseline variance.")

        # Two-column breakdown of top drivers and protective factors
        col_drivers, col_protect = st.columns(2)

        with col_drivers:
            st.markdown("<p style='font-size:0.82rem; font-weight:700; color:#E06D75; margin-bottom:0.5rem;'>🔴 Factors Increasing Churn Risk (+)</p>", unsafe_allow_html=True)
            if shap_res["risk_drivers"]:
                for feat_name, feat_val, delta_pct in shap_res["risk_drivers"][:5]:
                    st.markdown(f"""
                    <div class="obs-shap-row">
                        <div>
                            <span class="obs-shap-feat">{feat_name}</span>
                            <span class="obs-shap-val">({feat_val})</span>
                        </div>
                        <span class="obs-shap-delta" style="color:#E06D75;">+{delta_pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size:0.78rem; color:#717D96;'>No dominant factors pushing churn risk upward.</p>", unsafe_allow_html=True)

        with col_protect:
            st.markdown("<p style='font-size:0.82rem; font-weight:700; color:#69D6BE; margin-bottom:0.5rem;'>🟢 Protective Factors / Reducing Churn Risk (-)</p>", unsafe_allow_html=True)
            if shap_res["protective_factors"]:
                for feat_name, feat_val, delta_pct in shap_res["protective_factors"][:5]:
                    st.markdown(f"""
                    <div class="obs-shap-row">
                        <div>
                            <span class="obs-shap-feat">{feat_name}</span>
                            <span class="obs-shap-val">({feat_val})</span>
                        </div>
                        <span class="obs-shap-delta" style="color:#69D6BE;">{delta_pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size:0.78rem; color:#717D96;'>No dominant protective factors identified.</p>", unsafe_allow_html=True)

        st.markdown(f"""
        <p class="obs-footnote">
            <strong>Reconciliation & Integrity:</strong> Summing all 19 operational feature shifts ({sum(shap_res['grouped_shap'].values()):+.1f}% points)
            with the ensemble prior ({base_val:.1f}%) reconciles exactly with the model's output ({pct_base:.1f}%).
            Attributions dynamically update whenever any field in the profile is altered.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────
        # GLOBAL FEATURE IMPORTANCE & PRESCRIPTIVE RETENTION STRATEGIES
        # ─────────────────────────────────────────────────────────────
        col_fi, col_rec = st.columns(2)

        with col_fi:
            st.markdown('<div class="obs-section-card" style="height:100%;"><div class="obs-section-header">'
                        '<div>'
                        '<p class="obs-section-title">What Matters Across All Customers</p>'
                        '<p class="obs-section-caption">Global Gini Feature Importance across the trained Random Forest ensemble</p>'
                        '</div>'
                        '</div>', unsafe_allow_html=True)

            fi_chart, df_imp = build_global_importance_chart(baseline_pipe, top_n=7)
            if fi_chart is not None:
                st.altair_chart(fi_chart, use_container_width=True)
            else:
                st.info("Global feature importance is currently unavailable.")

            st.markdown("""
            <p class="obs-footnote">
                <strong>Crucial Distinction:</strong> Global Feature Importance measures overall impurity reduction across
                the entire training cohort (7,043 accounts). These metrics describe overall model structure,
                <strong>do NOT sum to any customer's churn probability</strong>, and remain constant across all accounts.
            </p>
            </div>
            """, unsafe_allow_html=True)

        with col_rec:
            st.markdown('<div class="obs-section-card" style="height:100%;"><div class="obs-section-header">'
                        '<p class="obs-section-title">Prescriptive Retention Considerations</p>'
                        '<p class="obs-section-caption">Targeted actions informed by observed dataset associations</p>'
                        '</div>', unsafe_allow_html=True)

            recs = []
            if contract == "Month-to-month":
                recs.append((
                    "Contract Term Stabilization",
                    "Historical Cohort Observation: Month-to-month accounts had a 42.7% churn rate in the cohort versus 11.3% for 1-year terms. "
                    "Actionable Step: Present a 1-year contract extension incentive or introductory term rate."
                ))
            if payment == "Electronic check":
                recs.append((
                    "Autopay & Billing Migration",
                    "Historical Cohort Observation: Electronic check users demonstrated the highest churn rate (45.3%) among all payment options. "
                    "Actionable Step: Incentivize migration to automatic ACH or card autopay with a recurring billing credit."
                ))
            if internet_service == "Fiber optic" and tech_support in ("No", "No internet service"):
                recs.append((
                    "Technical Support Trial",
                    "Historical Cohort Observation: High-speed Fiber subscribers without active Tech Support exhibited elevated churn. "
                    "Actionable Step: Provide a 60-day complimentary technical support onboarding package."
                ))
            if tenure <= 6:
                recs.append((
                    "Early-Tenure Onboarding",
                    "Historical Cohort Observation: Accounts with tenure under 6 months sit in the highest hazard window. "
                    "Actionable Step: Initiate proactive 30/60/90-day customer success onboarding touchpoints."
                ))

            if not recs:
                recs.append((
                    "Relationship Maintenance & Recognition",
                    "Historical Cohort Observation: Account parameters align with low-churn baseline profiles. "
                    "Actionable Step: Maintain high SLA standards and acknowledge account tenure milestones."
                ))

            for title, detail in recs[:3]:
                st.markdown(f"""
                <div style="background:#182236; border-left:3px solid #A69BFF; padding:0.7rem 0.95rem; border-radius:4px; margin-bottom:0.7rem;">
                    <div style="font-size:0.82rem; font-weight:700; color:#F1F3FA; margin-bottom:0.2rem;">{title}</div>
                    <div style="font-size:0.785rem; color:#A3AEC2; line-height:1.4;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <p class="obs-footnote">
                <strong>Methodological Boundary:</strong> Retention recommendations are strategic suggestions based on observed cohort correlations.
                They do not represent guaranteed retention improvements or causal guarantees.
            </p>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2: EXPERIMENTAL TELEMETRY LAB (STATUS & SATISFACTION)
    # ─────────────────────────────────────────────────────────────────
    with tab_experimental:
        st.markdown("""
        <div style="background: rgba(240, 201, 135, 0.08); border: 1px solid rgba(240, 201, 135, 0.28); border-radius: 6px; padding: 0.95rem 1.25rem; margin-bottom: 1.25rem; font-size: 0.835rem; line-height: 1.5; color: #F1F3FA;">
            <strong style="color:#F0C987;">🔬 Experimental Lab & Target Leakage Advisory:</strong><br>
            This laboratory benchmarks predictions from the experimental model (<code>models_expanded/best_churn_model_expanded.joblib</code>)
            which incorporates <code>Satisfaction Score</code> from <code>Telco_customer_churn_status.xlsx</code>.<br>
            In real-world telecom operations, customer satisfaction surveys are frequently collected post-cancellation or during exit workflows,
            introducing severe <strong>retrospective target leakage</strong>.
            The baseline model in Tab 1 remains the operational benchmark; this tab is provided strictly for academic and experimental inspection.
        </div>
        """, unsafe_allow_html=True)

        if expanded_pipe is None:
            st.warning("Expanded experimental model is not found in `models_expanded/`. Run `python -m src.run_experiments` to generate it.")
        else:
            col_bench, col_sim = st.columns([1, 1.2])

            with col_bench:
                st.markdown('<div class="obs-section-card" style="height:100%;"><div class="obs-section-header">'
                            '<p class="obs-section-title">Model Architecture Comparison</p>'
                            '<p class="obs-section-caption">Held-out test set performance (Identical 80/20 splits)</p>'
                            '</div>', unsafe_allow_html=True)

                st.markdown("""
                <table style="width:100%; border-collapse:collapse; font-size:0.82rem; color:#F1F3FA; text-align:left;">
                    <thead>
                        <tr style="border-bottom:1px solid #253248; color:#A3AEC2;">
                            <th style="padding:0.45rem 0;">Metric</th>
                            <th style="padding:0.45rem;">Baseline Model</th>
                            <th style="padding:0.45rem;">Experimental Model</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Algorithm</td>
                            <td style="padding:0.45rem;">Tuned Random Forest</td>
                            <td style="padding:0.45rem; color:#A69BFF; font-weight:600;">Tuned Random Forest</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Features</td>
                            <td style="padding:0.45rem;">19 Operational</td>
                            <td style="padding:0.45rem;">20 (+Satisfaction)</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Test Accuracy</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono';">75.44%</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono'; color:#69D6BE;">94.46%</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Test Recall</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono';">80.21%</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono'; color:#69D6BE;">90.37%</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Test Precision</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono';">52.45%</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono'; color:#69D6BE;">88.95%</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(37,50,72,0.5);">
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Test F1-Score</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono';">0.6342</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono'; color:#69D6BE;">0.8966</td>
                        </tr>
                        <tr>
                            <td style="padding:0.45rem 0; color:#CBD5E1;">Test ROC-AUC</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono';">0.8395</td>
                            <td style="padding:0.45rem; font-family:'JetBrains Mono'; color:#69D6BE;">0.9794</td>
                        </tr>
                    </tbody>
                </table>
                <p class="obs-footnote">
                    * The expanded model gains ~26 points in F1 primarily due to the strong retrospective correlation of Satisfaction Score.
                </p>
                </div>
                """, unsafe_allow_html=True)

            with col_sim:
                st.markdown('<div class="obs-section-card" style="height:100%;"><div class="obs-section-header">'
                            '<p class="obs-section-title">Telemetry Simulator & Delta Analysis</p>'
                            '<p class="obs-section-caption">Simulate how experimental model alters risk for this customer</p>'
                            '</div>', unsafe_allow_html=True)

                st.markdown("""
                <p style="font-size:0.82rem; color:#A3AEC2; margin-bottom:0.65rem;">
                    Select an experimental Satisfaction Score (1 = Lowest recorded, 5 = Highest recorded):
                </p>
                """, unsafe_allow_html=True)

                sim_score = st.radio(
                    "Simulated Score",
                    options=[1, 2, 3, 4, 5],
                    index=2,
                    horizontal=True,
                    format_func=lambda s: f"Score {s}",
                    key="exp_sim_score",
                    label_visibility="collapsed",
                )

                # Predict with experimental model
                exp_customer_dict = dict(full_feature_dict)
                exp_customer_dict["Satisfaction Score"] = sim_score
                exp_df = pd.DataFrame([exp_customer_dict])

                prob_exp = float(expanded_pipe.predict_proba(exp_df)[0, 1]) * 100
                delta = prob_exp - pct_base

                delta_color = "#E06D75" if delta > 0 else "#69D6BE"
                delta_sign = "+" if delta > 0 else ""

                st.markdown(f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.65rem; margin:0.85rem 0;">
                    <div style="background:#182236; border:1px solid #253248; border-radius:6px; padding:0.75rem; text-align:center;">
                        <div style="font-size:0.725rem; color:#A3AEC2; text-transform:uppercase;">Baseline Probability</div>
                        <div style="font-family:'JetBrains Mono'; font-size:1.6rem; font-weight:700; color:#F1F3FA;">{pct_base:.1f}%</div>
                    </div>
                    <div style="background:#182236; border:1px solid #253248; border-radius:6px; padding:0.75rem; text-align:center;">
                        <div style="font-size:0.725rem; color:#A3AEC2; text-transform:uppercase;">Experimental (Score {sim_score})</div>
                        <div style="font-family:'JetBrains Mono'; font-size:1.6rem; font-weight:700; color:{delta_color};">{prob_exp:.1f}%</div>
                        <div style="font-size:0.725rem; color:{delta_color};">Delta: {delta_sign}{delta:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Sensitivity Analysis Section
            st.markdown('<div class="obs-section-card"><div class="obs-section-header">'
                        '<p class="obs-section-title">Model Sensitivity Curve (Scores 1 to 5)</p>'
                        '<p class="obs-section-caption">Simulated experimental probability across all satisfaction ratings</p>'
                        '</div>', unsafe_allow_html=True)

            sensitivity_data = []
            for s_val in [1, 2, 3, 4, 5]:
                temp_dict = dict(full_feature_dict)
                temp_dict["Satisfaction Score"] = s_val
                s_prob = float(expanded_pipe.predict_proba(pd.DataFrame([temp_dict]))[0, 1]) * 100
                sensitivity_data.append((s_val, s_prob))

            for s_val, s_prob in sensitivity_data:
                is_active = (s_val == sim_score)
                fill_color = "#E06D75" if s_prob >= 65 else "#F0C987" if s_prob >= 35 else "#69D6BE"
                active_badge = f"<span style='color:#A69BFF; font-size:0.725rem; font-weight:700; margin-left:0.5rem;'>[Active Selection]</span>" if is_active else ""
                border_highlight = "border-color:#A69BFF;" if is_active else ""

                st.markdown(f"""
                <div class="obs-sens-row">
                    <div class="obs-sens-label">Score {s_val} {active_badge}</div>
                    <div class="obs-sens-track" style="{border_highlight}">
                        <div class="obs-sens-fill" style="width:{min(max(s_prob, 2.0), 100.0):.1f}%; background:{fill_color};"></div>
                    </div>
                    <div class="obs-sens-value" style="color:{fill_color};">{s_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <p class="obs-footnote">
                <strong>Methodological Disclaimer:</strong> Sensitivity analysis demonstrates how the experimental model
                evaluates hypothetical values for this profile. It does not prove that operational customer service improvements
                will causally change customer behavior by the simulated delta.
            </p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
