"""
dashboard/app.py
----------------
Interactive Streamlit dashboard for Customer Churn Analysis.
Run with: streamlit run dashboard/app.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))

import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; }
    .stMetric { background: #1c2333; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #e0e0e0; }
    .risk-high { color: #ef5350; font-weight: bold; }
    .risk-med  { color: #FFA726; font-weight: bold; }
    .risk-low  { color: #66BB6A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from data_loader import load_raw_data, clean_data
    from features import engineer_features, encode_features, prepare_features

    df = load_raw_data(config["data"]["raw_path"])
    df_clean = clean_data(df, config)
    return df_clean, config


@st.cache_resource
def load_model(model_path: str):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_churn(df_clean, model, config):
    from features import engineer_features, encode_features, prepare_features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df_clean, config)

    df_fe = engineer_features(df_clean)
    df_enc = encode_features(df_fe, config)
    drop_cols = config["features"]["drop_cols"] + [config["data"]["target_column"]]
    X_full = df_enc.drop(columns=drop_cols, errors="ignore")
    X_full = X_full.reindex(columns=feature_names, fill_value=0)

    probs = model.predict_proba(X_full)[:, 1]
    return probs


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/decrease.png", width=60)
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 EDA", "🤖 Model Performance", "🛡️ Retention Strategy", "🔮 Predict Customer"]
)

df_clean, config = load_data()
model_path = os.path.join(os.path.dirname(__file__), "..", config["model"]["model_path"])
model = load_model(model_path)

if model is not None:
    probs = predict_churn(df_clean, model, config)
    df_clean["churn_prob"] = probs

    high_t = config["retention"]["high_risk_threshold"]
    med_t  = config["retention"]["medium_risk_threshold"]

    def get_tier(p):
        if p >= high_t: return "🔴 High Risk"
        elif p >= med_t: return "🟡 Medium Risk"
        else: return "🟢 Low Risk"

    df_clean["risk_tier"] = df_clean["churn_prob"].apply(get_tier)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📉 Customer Churn Analysis Dashboard")
    st.markdown("---")

    total = len(df_clean)
    churned = df_clean["Churn"].sum()
    retained = total - churned

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}", f"{churned/total*100:.1f}%")
    c3.metric("Retained", f"{retained:,}", f"{retained/total*100:.1f}%")
    if model is not None:
        high_risk = (df_clean["churn_prob"] >= high_t).sum()
        c4.metric("🔴 High Risk", f"{high_risk:,}", "Need immediate action")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=[churned, retained],
            names=["Churned", "Retained"],
            color_discrete_sequence=["#ef5350", "#66BB6A"],
            title="Overall Churn Split",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_contract = df_clean.groupby("Contract")["Churn"].mean() * 100
        fig = px.bar(
            x=churn_contract.index, y=churn_contract.values,
            labels={"x": "Contract Type", "y": "Churn Rate (%)"},
            title="Churn Rate by Contract Type",
            color=churn_contract.values,
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    feature = st.selectbox(
        "Select categorical feature",
        ["Contract", "InternetService", "PaymentMethod", "gender",
         "SeniorCitizen", "Partner", "Dependents", "TechSupport"]
    )

    col1, col2 = st.columns(2)
    with col1:
        churn_by = df_clean.groupby(feature)["Churn"].mean().reset_index()
        churn_by.columns = [feature, "Churn Rate"]
        churn_by["Churn Rate"] *= 100
        fig = px.bar(churn_by, x=feature, y="Churn Rate",
                     title=f"Churn Rate by {feature}",
                     color="Churn Rate", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df_clean, x="MonthlyCharges",
                           color="Churn", barmode="overlay",
                           color_discrete_map={0: "#66BB6A", 1: "#ef5350"},
                           title="Monthly Charges Distribution by Churn",
                           labels={"Churn": "Churned"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    fig = px.box(df_clean, x="Contract", y="tenure",
                 color="Churn",
                 color_discrete_map={0: "#42A5F5", 1: "#ef5350"},
                 title="Tenure Distribution by Contract & Churn")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model not found. Run `python src/model.py` first.")
    else:
        metrics_path = os.path.join(os.path.dirname(__file__), "..", "reports", "model_metrics.csv")
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            st.dataframe(df_metrics.style.highlight_max(subset=["roc_auc","f1"], color="#1b5e20"), use_container_width=True)

        img_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        col1, col2 = st.columns(2)
        for fname, col, caption in [
            ("roc_curves.png", col1, "ROC Curves"),
            ("confusion_matrix.png", col2, "Confusion Matrix — LightGBM"),
        ]:
            fpath = os.path.join(img_dir, fname)
            if os.path.exists(fpath):
                col.image(fpath, caption=caption, use_column_width=True)

        for fname, caption in [
            ("shap_summary.png", "SHAP Summary — Feature Impact"),
            ("shap_importance.png", "SHAP Feature Importance"),
        ]:
            fpath = os.path.join(img_dir, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=caption)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — RETENTION STRATEGY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Retention Strategy":
    st.title("🛡️ Retention Strategy by Risk Tier")
    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model not found. Run `python src/model.py` first.")
    else:
        tier_summary = df_clean.groupby("risk_tier").agg(
            Customers=("churn_prob", "count"),
            Avg_Churn_Prob=("churn_prob", "mean"),
            Avg_Monthly_Charge=("MonthlyCharges", "mean"),
        ).round(3)
        st.dataframe(tier_summary, use_container_width=True)

        fig = px.pie(
            df_clean["risk_tier"].value_counts().reset_index(),
            names="risk_tier", values="count",
            color_discrete_map={
                "🔴 High Risk": "#ef5350",
                "🟡 Medium Risk": "#FFA726",
                "🟢 Low Risk": "#66BB6A"
            },
            title="Risk Tier Distribution",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Recommended Actions")
        col1, col2, col3 = st.columns(3)
        col1.error("**🔴 High Risk**\n\n- Immediate phone/email outreach\n- 15–20% discount for 1-year contract\n- Dedicated account manager\n- Escalate tech issues")
        col2.warning("**🟡 Medium Risk**\n\n- Proactive check-in within 7 days\n- Loyalty rewards & bundle offers\n- Beta feature invitations\n- Personalized usage report")
        col3.success("**🟢 Low Risk**\n\n- Monthly newsletter\n- Upsell streaming/security add-ons\n- Referral rewards program\n- NPS feedback collection")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT SINGLE CUSTOMER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Customer":
    st.title("🔮 Predict Churn for a Single Customer")
    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model not found. Run `python src/model.py` first.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            senior = st.selectbox("Senior Citizen", [0, 1])

        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        if st.button("🔍 Predict Churn Probability", type="primary"):
            from data_loader import clean_data
            from features import engineer_features, encode_features, prepare_features

            # Build a one-row dataframe matching the raw data schema
            row = {
                "customerID": "DEMO001",
                "gender": gender, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_security, "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": tech_support,
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
                "Churn": 0,  # placeholder
            }
            df_row = pd.DataFrame([row])
            df_row["TotalCharges"] = pd.to_numeric(df_row["TotalCharges"], errors="coerce")
            df_row["Churn"] = 0

            _, _, _, _, feature_names = prepare_features(df_clean, config)

            df_fe = engineer_features(df_row)
            df_enc = encode_features(df_fe, config)
            drop_cols = config["features"]["drop_cols"] + [config["data"]["target_column"]]
            X_pred = df_enc.drop(columns=drop_cols, errors="ignore")
            X_pred = X_pred.reindex(columns=feature_names, fill_value=0)

            prob = model.predict_proba(X_pred)[0][1]

            if prob >= high_t:
                tier, color = "🔴 HIGH RISK", "red"
                action = "Immediate outreach required. Offer contract upgrade + discount."
            elif prob >= med_t:
                tier, color = "🟡 MEDIUM RISK", "orange"
                action = "Proactive check-in within 7 days. Offer loyalty rewards."
            else:
                tier, color = "🟢 LOW RISK", "green"
                action = "Routine engagement. Consider upsell opportunities."

            st.markdown(f"## Churn Probability: **{prob*100:.1f}%**")
            st.markdown(f"### Risk Tier: **:{color}[{tier}]**")
            st.info(f"**Recommended Action:** {action}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef5350" if prob > 0.5 else "#FFA726"},
                    "steps": [
                        {"range": [0, 40], "color": "#1b5e20"},
                        {"range": [40, 70], "color": "#e65100"},
                        {"range": [70, 100], "color": "#b71c1c"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
