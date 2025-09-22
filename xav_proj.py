import streamlit as st
import numpy as np
import joblib

# ------------------------
# Load Preprocessing & Models
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
st.set_page_config(page_title="Credit Risk Dashboard", layout="centered")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home", "Prediction"])

# ------------------------
# Home Page
# ------------------------
if page == "Home":
    st.title("📊 Credit Risk Prediction Dashboard")
    st.write(
        """
        Welcome to the Credit Risk Prediction App!  

        This tool helps assess the **likelihood of loan default** based on financial details.  
        - Choose a model (CatBoost, XGBoost, or Random Forest)  
        - Enter loan details and financial ratios  
        - Get a prediction of whether the loan is **risky or safe**, along with probability  

        👉 Use the sidebar to switch to the **Prediction** page.
        """
    )

# ------------------------
# Prediction Page
# ------------------------
elif page == "Prediction":
    st.title("🔮 Loan Default Prediction")

    # Model selection
    model_choice = st.sidebar.radio("Choose a Model", list(models.keys()))

    # Input form
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

    # Prediction
    if submitted:
        # Ensure input order matches training numeric_cols
        input_data = np.array([[
            interest_rate,
            principal,
            cancelled,
            undisbursed,
            disbursed,
            repaid,
            due,
            exchange_adj,
            obligation,
            loans_held,
            repayment_ratio,
            loan_to_gdp
        ]])

        input_scaled = scaler.transform(imputer.transform(input_data))

        model = models[model_choice]
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = "⚠️ Risky Loan (Default Likely)" if prob > 0.5 else "✅ Safe Loan (Low Risk)"

        st.success(f"**Prediction ({model_choice}):** {prediction}")
        st.info(f"Predicted Default Probability: {prob:.2f}")
