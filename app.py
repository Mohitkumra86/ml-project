"""
Telecom Customer Churn Prediction – Streamlit Web Application
==============================================================
A professional, interactive web app that lets users input customer
details and get real-time churn predictions from the trained ML
pipeline.

Run:  streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS – dark modern dashboard theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Global overrides ──────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* ── Hero header ───────────────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.12) 100%);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 1.2rem;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.55);
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* ── Section cards ─────────────────────────────────────────── */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card-header {
        font-size: 1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(102,126,234,0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-icon {
        font-size: 1.15rem;
    }

    /* ── Result cards ──────────────────────────────────────────── */
    .result-card {
        border-radius: 1rem;
        padding: 1.6rem 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .card-high {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
    }
    .card-high::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .card-medium {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.3);
    }
    .card-medium::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .card-low {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
    }
    .card-low::before { background: linear-gradient(90deg, #10b981, #059669); }
    .card-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .card-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: rgba(255,255,255,0.92);
        margin: 0.2rem 0;
    }

    /* ── Probability gauge ─────────────────────────────────────── */
    .gauge-container {
        background: rgba(255,255,255,0.05);
        border-radius: 1rem;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .gauge-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }
    .gauge-track {
        background: rgba(255,255,255,0.08);
        border-radius: 0.5rem;
        height: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 0.5rem;
        transition: width 0.8s ease;
    }
    .gauge-fill-low    { background: linear-gradient(90deg, #10b981, #34d399); }
    .gauge-fill-medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .gauge-fill-high   { background: linear-gradient(90deg, #ef4444, #f87171); }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.35rem;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3);
        font-weight: 500;
    }

    /* ── Explanation / insight cards ────────────────────────────── */
    .insight-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 1rem;
        padding: 1.4rem 1.5rem;
        margin: 1rem 0;
    }
    .insight-header {
        font-size: 1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.85);
        margin-bottom: 0.3rem;
    }
    .insight-caption {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.4);
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .recommendation-card {
        border-radius: 0.8rem;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .rec-high {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        color: rgba(255,255,255,0.8);
    }
    .rec-medium {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        color: rgba(255,255,255,0.8);
    }
    .rec-low {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.25);
        color: rgba(255,255,255,0.8);
    }
    .rec-title {
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }

    /* ── Sidebar styling ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131326 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-icon {
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .sidebar-brand-name {
        font-size: 1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.85);
        letter-spacing: -0.01em;
    }
    .sidebar-brand-tag {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.35);
        font-weight: 500;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: rgba(255,255,255,0.35);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.2rem 0 0.7rem 0;
    }
    .kpi-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 0.7rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
        font-weight: 500;
    }
    .kpi-value {
        font-size: 1rem;
        font-weight: 800;
        color: rgba(255,255,255,0.9);
    }
    .kpi-value-accent {
        font-size: 1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-info-card {
        background: rgba(102,126,234,0.08);
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 0.7rem;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
    }
    .sidebar-info-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.4);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-info-value {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.85);
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .risk-legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        font-size: 0.84rem;
        color: rgba(255,255,255,0.6);
    }
    .risk-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .risk-dot-high   { background: #ef4444; }
    .risk-dot-medium { background: #f59e0b; }
    .risk-dot-low    { background: #10b981; }

    /* ── Footer ────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255,255,255,0.2);
        font-size: 0.78rem;
        font-weight: 500;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2.5rem;
    }
    .app-footer span {
        color: rgba(102,126,234,0.5);
    }

    /* ── Button styling ────────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        padding: 0.85rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 0.8rem;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Hide Streamlit branding ───────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Feature bar chart ─────────────────────────────────────── */
    .feat-bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.45rem;
        gap: 0.6rem;
    }
    .feat-bar-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.6);
        min-width: 180px;
        text-align: right;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .feat-bar-track {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 0.3rem;
        height: 0.7rem;
        overflow: hidden;
    }
    .feat-bar-fill {
        height: 100%;
        border-radius: 0.3rem;
        background: linear-gradient(90deg, #667eea, #a78bfa);
    }
    .feat-bar-value {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.45);
        min-width: 42px;
        font-weight: 600;
    }

    /* ── Selectbox/input styling in dark theme ─────────────────── */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: rgba(255,255,255,0.6) !important;
    }

    /* ── Responsive tweaks ─────────────────────────────────────── */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.6rem; }
        .hero-container { padding: 1.5rem; }
        .card-value { font-size: 1.8rem; }
        .feat-bar-label { min-width: 120px; font-size: 0.7rem; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Load model (cached so it loads only once)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__),
                              "models", "best_churn_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


pipeline = load_model()


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">📡</div>
        <div class="sidebar-brand-name">Churn Predictor</div>
        <div class="sidebar-brand-tag">ML-Powered Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset info
    st.markdown('<div class="sidebar-section-title">Dataset</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Source</div>
        <div class="sidebar-info-value">Telco Customer Churn</div>
    </div>
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Customers</div>
        <div class="sidebar-info-value">7,043</div>
    </div>
    """, unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="sidebar-section-title">Model</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Algorithm</div>
        <div class="sidebar-info-value">Tuned Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI metrics
    st.markdown('<div class="sidebar-section-title">Performance</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="kpi-card">
        <span class="kpi-label">ROC-AUC</span>
        <span class="kpi-value-accent">0.840</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label">Recall</span>
        <span class="kpi-value-accent">0.802</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label">F1 Score</span>
        <span class="kpi-value-accent">0.634</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label">Precision</span>
        <span class="kpi-value">0.524</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label">Accuracy</span>
        <span class="kpi-value">0.754</span>
    </div>
    """, unsafe_allow_html=True)

    # Risk legend
    st.markdown('<div class="sidebar-section-title">Risk Levels</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="risk-legend-item">
        <span class="risk-dot risk-dot-high"></span> High — probability ≥ 75%
    </div>
    <div class="risk-legend-item">
        <span class="risk-dot risk-dot-medium"></span> Medium — probability 45–75%
    </div>
    <div class="risk-legend-item">
        <span class="risk-dot risk-dot-low"></span> Low — probability < 45%
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <p class="hero-title">Telecom Customer Churn Prediction</p>
    <p class="hero-subtitle">
        Predict customer churn risk using a trained Random Forest model.
        Enter customer details below and get an instant risk assessment
        powered by machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Input form — three column sections with card wrappers
# ─────────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns(3, gap="large")

def _sync_total_charges():
    t = st.session_state.get("tenure_slider", 12)
    m = st.session_state.get("monthly_charges_input", 70.0)
    st.session_state["total_charges_input"] = min(max(float(round(t * m, 2)), 0.0), 9000.0)

if "total_charges_input" not in st.session_state:
    init_t = st.session_state.get("tenure_slider", 12)
    init_m = st.session_state.get("monthly_charges_input", 70.0)
    st.session_state["total_charges_input"] = min(max(float(round(init_t * init_m, 2)), 0.0), 9000.0)

# ── Section 1: Customer Profile ─────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="section-card-header">
        <span class="section-icon">👤</span> Customer Profile
    </div>
    """, unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure_slider", on_change=_sync_total_charges)

# ── Section 2: Services ─────────────────────────────────────────────
with col_mid:
    st.markdown("""
    <div class="section-card-header">
        <span class="section-icon">🌐</span> Services
    </div>
    """, unsafe_allow_html=True)

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"
        st.info("Multiple Lines: *No phone service*")

    internet_service = st.selectbox(
        "Internet Service", ["Fiber optic", "DSL", "No"])

    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.info("Add-on services: *No internet service*")

# ── Section 3: Contract & Billing ────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="section-card-header">
        <span class="section-icon">💳</span> Contract & Billing
    </div>
    """, unsafe_allow_html=True)

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"])

    monthly_charges = st.number_input(
        "Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5, key="monthly_charges_input", on_change=_sync_total_charges)
    total_charges = st.number_input(
        "Total Charges ($)", 0.0, 9000.0, step=10.0, key="total_charges_input")


# ─────────────────────────────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔮  Predict Churn Risk",
                                use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Build feature dict & predict
# ─────────────────────────────────────────────────────────────────────
if predict_clicked:
    customer = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    # Build DataFrame in exact column order
    feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    df = pd.DataFrame([customer], columns=feature_cols)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    prob = pipeline.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    # Risk level
    if prob >= 0.75:
        risk, risk_css = "HIGH", "card-high"
        risk_icon = "🔴"
        gauge_css = "gauge-fill-high"
        rec_css = "rec-high"
    elif prob >= 0.45:
        risk, risk_css = "MEDIUM", "card-medium"
        risk_icon = "🟡"
        gauge_css = "gauge-fill-medium"
        rec_css = "rec-medium"
    else:
        risk, risk_css = "LOW", "card-low"
        risk_icon = "🟢"
        gauge_css = "gauge-fill-low"
        rec_css = "rec-low"

    # ── Section 4: Results ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3, gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-card {risk_css}">
            <p class="card-label">Churn Probability</p>
            <p class="card-value">{prob * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        pred_text = "WILL CHURN" if pred == 1 else "WILL NOT CHURN"
        pred_icon = "⚠️" if pred == 1 else "✅"
        st.markdown(f"""
        <div class="result-card {risk_css}">
            <p class="card-label">Prediction</p>
            <p class="card-value" style="font-size:1.5rem;">{pred_icon} {pred_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="result-card {risk_css}">
            <p class="card-label">Risk Level</p>
            <p class="card-value">{risk_icon} {risk} RISK</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability gauge bar ───────────────────────────────────────
    prob_pct = prob * 100
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-label">Churn Probability Indicator</div>
        <div class="gauge-track">
            <div class="gauge-fill {gauge_css}" style="width: {prob_pct:.1f}%;"></div>
        </div>
        <div class="gauge-labels">
            <span>0% — Low Risk</span>
            <span>45%</span>
            <span>75%</span>
            <span>100% — High Risk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Part A: Model Feature Importance (native Streamlit) ──────
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    importances = classifier.feature_importances_
    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names_clean = [
        n.replace("num__", "").replace("cat__", "") for n in feature_names_raw
    ]

    # Sort by global importance and take top 5
    importance_data = sorted(
        zip(feature_names_clean, importances),
        key=lambda x: x[1], reverse=True
    )[:5]
    max_imp = importance_data[0][1] if importance_data else 1

    st.markdown("---")
    st.markdown("#### 🧠 What Drove This Prediction")
    st.caption(
        "These values represent the Random Forest model's learned **global "
        "feature importance**. They indicate which features were most "
        "influential to the model overall, not a guaranteed causal "
        "explanation for this individual customer."
    )

    for feat_name, feat_imp in importance_data:
        pct = feat_imp * 100
        bar_value = feat_imp / max_imp  # normalise to [0, 1] for st.progress
        col_label, col_bar, col_val = st.columns([3, 6, 1])
        with col_label:
            st.markdown(
                f"<span style='font-size:0.85rem; color:rgba(255,255,255,0.7); "
                f"font-weight:500;'>{feat_name}</span>",
                unsafe_allow_html=True,
            )
        with col_bar:
            st.progress(bar_value)
        with col_val:
            st.markdown(
                f"<span style='font-size:0.82rem; color:rgba(255,255,255,0.5); "
                f"font-weight:600;'>{pct:.1f}%</span>",
                unsafe_allow_html=True,
            )

    # ── Part B: Dataset-Level Churn Insights (context) ────────────
    risk_insights = []
    if contract == "Month-to-month":
        risk_insights.append(
            "Customers on <strong>month-to-month contracts</strong> had a "
            "42.7% churn rate in the dataset — the highest of all contract types.")
    if internet_service == "Fiber optic":
        risk_insights.append(
            "<strong>Fiber optic</strong> subscribers churned at roughly "
            "2× the rate of DSL subscribers in the dataset.")
    if payment == "Electronic check":
        risk_insights.append(
            "<strong>Electronic check</strong> was the payment method with "
            "the highest churn rate in the dataset.")
    if tenure <= 6:
        risk_insights.append(
            f"Customers with <strong>tenure ≤ 6 months</strong> (this customer: "
            f"{tenure}) had the highest churn rate in the dataset.")
    elif tenure <= 12:
        risk_insights.append(
            f"Customers with <strong>tenure ≤ 12 months</strong> (this customer: "
            f"{tenure}) churned at above-average rates in the dataset.")
    if senior == "Yes":
        risk_insights.append(
            "<strong>Senior citizens</strong> churned at roughly 2× the rate "
            "of non-seniors in the dataset.")
    if online_security == "No" and internet_service != "No":
        risk_insights.append(
            "Customers <strong>without online security</strong> had a higher "
            "churn rate than those with the service.")
    if tech_support == "No" and internet_service != "No":
        risk_insights.append(
            "Customers <strong>without tech support</strong> had a higher "
            "churn rate than those with the service.")
    if monthly_charges > 70:
        risk_insights.append(
            f"Higher <strong>monthly charges</strong> (this customer: "
            f"${monthly_charges:.0f}) were correlated with higher "
            f"churn in the dataset.")

    protective_insights = []
    if contract == "Two year":
        protective_insights.append(
            "<strong>Two-year contract</strong> customers had only a 2.8% "
            "churn rate in the dataset — the lowest.")
    elif contract == "One year":
        protective_insights.append(
            "<strong>One-year contract</strong> customers had an 11.3% "
            "churn rate — much lower than month-to-month.")
    if tenure >= 48:
        protective_insights.append(
            f"Customers with <strong>long tenure</strong> (this customer: "
            f"{tenure} months) rarely churned in the dataset.")
    if online_security == "Yes":
        protective_insights.append(
            "Customers <strong>with online security</strong> had a "
            "substantially lower churn rate in the dataset.")
    if tech_support == "Yes":
        protective_insights.append(
            "Customers <strong>with tech support</strong> had a lower "
            "churn rate in the dataset.")
    if payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        protective_insights.append(
            "<strong>Automatic payment methods</strong> were associated "
            "with lower churn rates in the dataset.")

    insights_html = ""
    if risk_insights:
        insights_html += '<div style="margin-bottom:0.7rem;font-weight:600;color:rgba(255,255,255,0.7);font-size:0.88rem;">Risk-associated patterns matching this customer</div>'
        for ins in risk_insights:
            insights_html += f'<div style="display:flex;gap:0.5rem;align-items:flex-start;margin-bottom:0.4rem;font-size:0.85rem;color:rgba(255,255,255,0.6);line-height:1.5;"><span style="color:#ef4444;">●</span><span>{ins}</span></div>'

    if protective_insights:
        insights_html += '<div style="margin-top:0.8rem;margin-bottom:0.7rem;font-weight:600;color:rgba(255,255,255,0.7);font-size:0.88rem;">Protective patterns matching this customer</div>'
        for ins in protective_insights:
            insights_html += f'<div style="display:flex;gap:0.5rem;align-items:flex-start;margin-bottom:0.4rem;font-size:0.85rem;color:rgba(255,255,255,0.6);line-height:1.5;"><span style="color:#10b981;">●</span><span>{ins}</span></div>'

    if not risk_insights and not protective_insights:
        insights_html = '<div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">This customer\'s profile does not strongly match any high-risk or protective patterns observed in the dataset.</div>'

    st.markdown(f"""
    <div class="insight-section">
        <div class="insight-header">📊 Key Churn Insights from the Dataset</div>
        <div class="insight-caption">
            General patterns observed across all 7,043 customers in the training
            data. These provide useful context but are <em>not</em> the model's
            reasoning for this individual prediction — see the chart above for that.
        </div>
        {insights_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendation ────────────────────────────────────────────
    if risk == "HIGH":
        rec_text = (
            "This customer has a <strong>high predicted churn risk</strong>. "
            "Consider offering a retention package — such as a discounted "
            "long-term contract, free add-on services, or a dedicated "
            "support agent — to reduce churn risk.")
    elif risk == "MEDIUM":
        rec_text = (
            "This customer shows <strong>moderate churn risk</strong>. "
            "Proactive engagement — a loyalty discount, service upgrade, "
            "or satisfaction survey — could help retain them.")
    else:
        rec_text = (
            "This customer has a <strong>low churn risk</strong>. "
            "Maintain current service quality and consider upselling "
            "additional services.")

    st.markdown(f"""
    <div class="recommendation-card {rec_css}">
        <div class="rec-title">💡 Recommended Action</div>
        <div>{rec_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built with <span>Python</span> · <span>Scikit-learn</span> ·
    <span>Random Forest</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)
