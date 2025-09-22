import streamlit as st
import numpy as np
import joblib
import pandas as pd

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
# Page Config
# ------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="💳")

# ------------------------
# Custom CSS for Background & Styling
# ------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1556740761-90f6e46b9b7f');
        background-size: cover;
        background-attachment: fixed;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stMarkdown h1 {
        color: #003366;
    }
    .stMarkdown p {
        font-size: 18px;
        color: #000000;
    }
    .metric-label {
        font-weight: bold;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Sidebar Navigation
# ------------------------
page = st.sidebar.radio("Navigate", ["Home", "Prediction", "Analytics"])

# ------------------------
# Home Page
# ------------------------
if page == "Home":
    st.title("💳 Credit Risk Prediction Dashboard")
    
    st.markdown(
        """
        Welcome to the **Credit Risk Prediction App**!  

        This app is designed to help **financial analysts and institutions** assess the likelihood of loan defaults quickly and efficiently.  

        ### Key Features:
        - ✅ **Multiple Models:** CatBoost, XGBoost, Random Forest
        - ✅ **Instant Predictions:** Risk classification with probability score
        - ✅ **Interactive Analytics:** Charts, metrics, and visual risk indicators
        - ✅ **User-Friendly Interface:** Easy input and navigation

        ### How to Use:
        1. Go to the **Prediction** page.
        2. Enter loan details and financial ratios.
        3. Select a model and click **Predict Risk**.
        4. View results and probability.
        """
    )

# ------------------------
# Prediction Page
# ------------------------
elif page == "Prediction":
    st.title("🔮 Loan Default Prediction")

    model_choice = st.sidebar.radio("Choose a Model", list(models.keys()))

    with st.form("input_form"):
        st.subheader("Loan Details Input")

        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
        principal = st.number_input("Original Principal Amount (USD)", min_value=0.0, value=100000.0)
        cancelled = st.number_input("Cancelled Amount (USD)", min_value=0.0, value=0.0)
        undisbursed = st.number_input("Undisbursed Amount (USD)", min_value=0.0, value=0.0)
        disbursed = st.number_input("Disbursed Amount (USD)", min_value=0.0, value=50000.0)
        repaid = st.number_input("Repaid to IBRD (USD)", min_value=0.0, value=20000.0)
        due = st.number_input("Due to IBRD (USD)", min_value=0.0, value=10000.0)
        exchange_adj = st.number_input("Exchange Adjustment (USD)", min_value=0.0, value=0.0)
        obligation = st.number_input("Borrower's Obligation (USD)", min_value=0.0, value=80000.0)
        loans_held = st.number_input("Loans Held (USD)", min_value=0.0, value=0.0)
        repayment_ratio = st.number_input("Repayment Ratio (Repaid ÷ Disbursed)", min_value=0.0, max_value=10.0, value=0.5)
        loan_to_gdp = st.number_input("Loan-to-GDP Growth Ratio", min_value=0.0, max_value=10.0, value=1.0)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = np.array([[
            interest_rate, principal, cancelled, undisbursed, disbursed,
            repaid, due, exchange_adj, obligation, loans_held,
            repayment_ratio, loan_to_gdp
        ]])

        input_scaled = scaler.transform(imputer.transform(input_data))
        model = models[model_choice]
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = "⚠️ Risky Loan (Default Likely)" if prob > 0.5 else "✅ Safe Loan (Low Risk)"

        # Show prediction with gauge bar
        st.success(f"**Prediction ({model_choice}):** {prediction}")
        st.progress(int(prob*100))
        st.info(f"Predicted Default Probability: {prob:.2f}")

# ------------------------
# Analytics Page
# ------------------------
elif page == "Analytics":
    st.title("📊 Interactive Analytics")

    st.markdown(
        """
        Explore how different financial ratios and loan features impact credit risk.
        Use the interactive sliders and charts below to simulate different scenarios.
        """
    )

    # Example interactive metrics
    interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    repayment_ratio = st.slider("Repayment Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    loan_to_gdp = st.slider("Loan-to-GDP Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    st.markdown("### Simulated Risk Probability")
    # Simple simulated formula for interactive demo
    simulated_prob = min(1.0, 0.3 + 0.03*interest_rate - 0.2*repayment_ratio + 0.05*loan_to_gdp)
    st.metric(label="Predicted Default Probability", value=f"{simulated_prob:.2f}")
    st.progress(int(simulated_prob*100))

    st.markdown("### Scenario Analysis")
    scenario_df = pd.DataFrame({
        "Feature": ["Interest Rate", "Repayment Ratio", "Loan-to-GDP"],
        "Value": [interest_rate, repayment_ratio, loan_to_gdp]
    })
    st.table(scenario_df)

    st.markdown("### Insights")
    with st.expander("Click to see interpretation"):
        st.write(
            """
            - Higher **interest rates** increase the risk of default.
            - Higher **repayment ratios** reduce default risk.
            - A higher **loan-to-GDP ratio** slightly increases risk depending on economic conditions.
            """
        )
