import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt

# ------------------------
# Load Models & Preprocessing
# ------------------------
catboost_model = joblib.load("catboost_model.pkl")
xgboost_model = joblib.load("xgboost_model.pkl")
rf_model = joblib.load("randomforest_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

models = {
    "CatBoost": catboost_model,
    "XGBoost": xgboost_model,
    "Random Forest": rf_model
}

# ------------------------
# Page Config & CSS
# ------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="💳")

st.markdown("""
<style>
.stApp {
    background-color: #0d0d0d;
    color: #f0f0f0;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #ffffff;
}
.stMarkdown p {
    font-size: 18px;
    color: #e0e0e0;
}
.stMetric label, .stMetric div {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Sidebar Navigation
# ------------------------
page = st.sidebar.radio("Navigate", ["Home", "Prediction", "Analytics"])

# ------------------------
# Home Page
# ------------------------
if page == "Home":
    st.title("💳 Credit Risk Prediction Dashboard")
    st.markdown("""
    Welcome to the **Credit Risk Prediction App**!  

    Assess the likelihood of loan defaults efficiently with interactive features and real-time predictions.  

    ### Key Features:
    - ✅ **Multiple Models:** CatBoost, XGBoost, Random Forest
    - ✅ **Interactive Analytics:** Simulate scenarios and visualize impact
    - ✅ **Animated Risk Indicators:** Gauge-style visualization
    """)

# ------------------------
# Prediction Page with Animated Gauge
# ------------------------
elif page == "Prediction":
    st.title("📉 Loan Default Prediction")

    model_choice = st.sidebar.radio("Choose a Model", list(models.keys()))

    with st.form("input_form"):
        st.subheader("Loan Details Input")
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 5.0)
        principal = st.number_input("Original Principal Amount (USD)", 0.0, 1e7, 1000000.0)
        cancelled = st.number_input("Cancelled Amount (USD)", 0.0, 1e7, 0.0)
        undisbursed = st.number_input("Undisbursed Amount (USD)", 0.0, 1e7, 0.0)
        disbursed = st.number_input("Disbursed Amount (USD)", 0.0, 1e7, 5000000.0)
        repaid = st.number_input("Repaid to IBRD (USD)", 0.0, 1e7, 20000.0)
        due = st.number_input("Due to IBRD (USD)", 0.0, 1e7, 1000000.0)
        exchange_adj = st.number_input("Exchange Adjustment (USD)", 0.0, 1e7, 0.0)
        obligation = st.number_input("Borrower's Obligation (USD)", 0.0, 1e7, 80000.0)
        loans_held = st.number_input("Loans Held (USD)", 0.0, 1e7, 0.0)
        repayment_ratio = st.number_input("Repayment Ratio", 0.0, 10.0, 0.5)
        loan_to_gdp = st.number_input("Loan-to-GDP Growth Ratio", 0.0, 10.0, 1.0)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = np.array([[interest_rate, principal, cancelled, undisbursed, disbursed,
                                repaid, due, exchange_adj, obligation, loans_held,
                                repayment_ratio, loan_to_gdp]])
        input_scaled = scaler.transform(imputer.transform(input_data))
        model = models[model_choice]

        with st.spinner("Analyzing loan risk..."):
            time.sleep(1)
            prob = model.predict_proba(input_scaled)[0][1]

        prediction = "⚡ Risky Loan (Default Likely)" if prob > 0.5 else "✅ Safe Loan (Low Risk)"
        st.success(f"**Prediction ({model_choice}):** {prediction}")

        # Animated circular gauge simulation
        gauge_placeholder = st.empty()
        for i in range(0, int(prob*100)+1, 2):
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie([i, 100-i], startangle=90, colors=['#FF4B4B','#444444'], wedgeprops={'width':0.3})
            ax.set(aspect="equal")
            gauge_placeholder.pyplot(fig)
            time.sleep(0.03)

        st.info(f"Predicted Default Probability: {prob:.2f}")

# ------------------------
# Analytics Page with Animated Probability
# ------------------------
elif page == "Analytics":
    st.title("📊 Interactive Analytics")
    st.markdown("Use sliders to simulate risk and watch the gauge animate dynamically.")

    interest_rate = st.slider("Interest Rate (%)", 0.0, 20.0, 5.0, 0.5)
    repayment_ratio = st.slider("Repayment Ratio", 0.0, 1.0, 0.5, 0.05)
    loan_to_gdp = st.slider("Loan-to-GDP Ratio", 0.0, 5.0, 1.0, 0.1)

    simulated_prob = min(1.0, 0.3 + 0.03*interest_rate - 0.2*repayment_ratio + 0.05*loan_to_gdp)

    st.markdown("### Simulated Risk Probability")
    gauge_placeholder = st.empty()
    for i in range(0, int(simulated_prob*100)+1, 2):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([i, 100-i], startangle=90, colors=['#00FFFF','#333333'], wedgeprops={'width':0.3})
        ax.set(aspect="equal")
        gauge_placeholder.pyplot(fig)
        time.sleep(0.02)

    st.metric(label="Predicted Default Probability", value=f"{simulated_prob:.2f}")

    st.markdown("### Scenario Analysis")
    scenario_df = pd.DataFrame({
        "Feature": ["Interest Rate", "Repayment Ratio", "Loan-to-GDP"],
        "Value": [interest_rate, repayment_ratio, loan_to_gdp]
    })
    st.table(scenario_df)

    with st.expander("Click to see insights"):
        st.write("""
        - Higher interest rates increase default risk.
        - Higher repayment ratios reduce risk.
        - Higher loan-to-GDP ratio slightly increases risk.
        """)
