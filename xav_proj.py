import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------
# Load Preprocessing & Models
# ------------------------
# Make sure to save your trained models and preprocessing objects in copy_of_pizza.py like:
# joblib.dump(best_model, "catboost_model.pkl")
# joblib.dump(best_model, "xgboost_model.pkl")
# joblib.dump(best_model, "randomforest_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(imputer, "imputer.pkl")

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
# Streamlit App
# ------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="centered")

st.title("📊 Credit Risk Prediction Dashboard")
st.write("Enter loan/financial details to predict the risk of default. Choose a model and get instant feedback.")

# Sidebar for model choice
model_choice = st.sidebar.radio("Choose a Model", list(models.keys()))

# Input form
with st.form("input_form"):
    st.subheader("Loan Details Input")

    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    principal = st.number_input("Original Principal Amount (USD)", min_value=0.0, value=100000.0)
    disbursed = st.number_input("Disbursed Amount (USD)", min_value=0.0, value=50000.0)
    repaid = st.number_input("Repaid to IBRD (USD)", min_value=0.0, value=20000.0)
    obligation = st.number_input("Borrower's Obligation (USD)", min_value=0.0, value=80000.0)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Feature vector
    input_data = np.array([[interest_rate, principal, disbursed, repaid, obligation]])
    input_scaled = scaler.transform(imputer.transform(input_data))

    # Predict
    model = models[model_choice]
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "⚠️ Risky Loan (Default Likely)" if prob > 0.5 else "✅ Safe Loan (Low Risk)"

    # Output
    st.success(f"**Prediction ({model_choice}):** {prediction}")
    st.info(f"Predicted Default Probability: {prob:.2f}")
