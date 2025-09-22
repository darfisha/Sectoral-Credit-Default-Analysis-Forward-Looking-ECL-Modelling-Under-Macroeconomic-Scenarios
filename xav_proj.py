# -*- coding: utf-8 -*-
"""Streamlit Credit Risk Analysis App."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import gdown

st.set_page_config(layout="wide", page_title="Credit Risk Analysis")

# ===============================
# Data Loading & Preprocessing
# ===============================
@st.cache_data
def load_and_preprocess_data():
    file_id = "1MVW1amhh9k3ksDsJkRo9ELvEwRplG0r2"   # replace with your dataset file id if needed
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "credit_risk.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    df = pd.read_csv(output, low_memory=False)

    # clean column names
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace("$", "usd")
        .replace("'", "").replace(".", "")
        for col in df.columns
    ]

    # filter India
    india_df = df[df['country___economy'].str.strip() == 'India'].copy()
    india_df.drop(columns=['currency_of_commitment'], inplace=True, errors='ignore')

    # numeric handling
    numeric_cols_raw = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd'
    ]
    india_df[numeric_cols_raw] = india_df[numeric_cols_raw].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols_raw:
        india_df[col] = india_df[col].apply(lambda x: np.nan if x < 0 else x)

    # default flag
    def encode_default_balanced(status, disbursed_amount):
        if not isinstance(status, str): return 1
        status = status.strip().upper()
        if status in ["FULLY REPAID", "SIGNED", "APPROVED", "DISBURSING"]: return 0
        if status in ["REPAYING", "DISBURSED", "DISBURSING&REPAYING", "FULLY DISBURSED"]: return 1
        if status in ["CANCELLED", "FULLY CANCELLED"]:
            return 1 if disbursed_amount and disbursed_amount > 0 else 0
        return 1
    india_df["default_flag"] = india_df.apply(
        lambda row: encode_default_balanced(row["loan_status"], row["disbursed_amount_ususd"]), axis=1
    )

    # feature engineering
    india_df["repayment_ratio"] = (
        india_df["repaid_to_ibrd_ususd"] / india_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    india_df["loan_to_gdp_growth_ratio"] = (
        india_df["original_principal_amount_ususd"] / 1e9
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return india_df

# ===============================
# Train Models
# ===============================
@st.cache_resource
def train_models(df):
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd',
        "repayment_ratio", "loan_to_gdp_growth_ratio"
    ]

    train_df = df[df["year"] <= 2020].copy()
    test_df  = df[df["year"] >= 2023].copy()

    X_train = train_df[numeric_cols].values
    y_train = train_df["default_flag"].values
    X_test  = test_df[numeric_cols].values if not test_df.empty else np.empty((0, len(numeric_cols)))
    y_test  = test_df["default_flag"].values if not test_df.empty else np.array([])

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed  = imputer.transform(X_test) if X_test.size else X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled  = scaler.transform(X_test_imputed) if X_test_imputed.size else X_test_imputed

    models = {
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                               min_samples_split=5, class_weight="balanced", random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if y_test.size > 0 else [0]
        auc = roc_auc_score(y_test, y_prob) if y_test.size > 0 else 0.0
        trained_models[name] = {"model": model, "test_auc": auc,
                                "scaler": scaler, "imputer": imputer}
    return trained_models

# ===============================
# Add Predictions
# ===============================
@st.cache_data
def get_all_data_with_predictions(df):
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd',
        "repayment_ratio", "loan_to_gdp_growth_ratio"
    ]

    models = train_models(df)
    model_info = models["CatBoost"]
    catboost_model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    X_all = df[numeric_cols].values
    X_all_scaled = scaler.transform(imputer.transform(X_all))

    df = df.copy()
    df['default_prob'] = catboost_model.predict_proba(X_all_scaled)[:, 1]
    return df

# ===============================
# Main App
# ===============================
st.markdown('<h1 class="main-header">Credit Risk Analysis & Prediction 📊</h1>', unsafe_allow_html=True)

with st.spinner("Loading data and training models..."):
    merged_df = load_and_preprocess_data()
    trained_models = train_models(merged_df)
    merged_df = get_all_data_with_predictions(merged_df)
    st.success("Loading complete!")

# Sidebar inputs
st.sidebar.header("Input Features 🚀")
input_features = {
    'interest_rate': st.sidebar.number_input("Interest Rate", value=0.05, format="%.2f"),
    'original_principal_amount_ususd': st.sidebar.number_input("Original Principal ($)", value=100_000_000, format="%d"),
    'cancelled_amount_ususd': st.sidebar.number_input("Cancelled Amount ($)", value=0, format="%d"),
    'undisbursed_amount_ususd': st.sidebar.number_input("Undisbursed Amount ($)", value=10_000_000, format="%d"),
    'disbursed_amount_ususd': st.sidebar.number_input("Disbursed Amount ($)", value=90_000_000, format="%d"),
    'repaid_to_ibrd_ususd': st.sidebar.number_input("Repaid to IBRD ($)", value=50_000_000, format="%d"),
    'due_to_ibrd_ususd': st.sidebar.number_input("Due to IBRD ($)", value=10_000_000, format="%d"),
    'exchange_adjustment_ususd': st.sidebar.number_input("Exchange Adjustment ($)", value=0, format="%d"),
    'borrowers_obligation_ususd': st.sidebar.number_input("Borrower's Obligation ($)", value=40_000_000, format="%d"),
    'loans_held_ususd': st.sidebar.number_input("Loans Held ($)", value=100_000_000, format="%d"),
    'gdp_growth': st.sidebar.number_input("GDP Growth (%)", value=5.5, format="%.2f"),
    'cpi_inflation': st.sidebar.number_input("CPI Inflation (%)", value=4.5, format="%.2f"),
}
selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model ✨",
    list(trained_models.keys()),
    index=0
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Individual Prediction", "Portfolio Analysis", "Stress Testing"])

# -------------------------------
# Tab 1 – Individual Prediction
# -------------------------------
with tab1:
    st.header("Individual Project Prediction")
    model_info = trained_models[selected_model_name]
    model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    # preprocess input
    input_df = pd.DataFrame([input_features])
    input_df["repayment_ratio"] = (
        input_df["repaid_to_ibrd_ususd"] / input_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    input_df["loan_to_gdp_growth_ratio"] = (
        input_df["original_principal_amount_ususd"] / (input_df["gdp_growth"] * 1e9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    input_features_list = input_df.values[0]
    input_imputed = imputer.transform([input_features_list])
    input_scaled = scaler.transform(input_imputed)
    pd_prob = model.predict_proba(input_scaled)[0, 1]

    # ECL
    usd_to_inr = 83
    LGD = 0.45
    EAD_INR = input_features['borrowers_obligation_ususd'] * usd_to_inr
    ECL_project = EAD_INR * LGD * pd_prob

    st.metric("Probability of Default (PD)", f"{pd_prob:.2%}")
    st.metric("Expected Credit Loss (ECL)", f"₹ {ECL_project:,.2f}")

# -------------------------------
# Tab 2 – Portfolio Analysis
# -------------------------------
with tab2:
    st.header("Portfolio & Sectoral Analysis")

    usd_to_inr = 83
    merged_df['borrowers_obligation_inr'] = merged_df['borrowers_obligation_ususd'] * usd_to_inr
    merged_df['repaid_to_ibrd_inr'] = merged_df['repaid_to_ibrd_ususd'] * usd_to_inr
    merged_df['LGD'] = ((merged_df['borrowers_obligation_inr'] - merged_df['repaid_to_ibrd_inr']) /
                        merged_df['borrowers_obligation_inr']).fillna(0)

    ecl_df = merged_df.groupby('project_name').apply(
        lambda x: pd.Series({
            'EAD_INR': x['borrowers_obligation_inr'].sum(),
            'PD': x['default_prob'].mean(),
            'LGD': (x['LGD'] * x['borrowers_obligation_inr']).sum() / x['borrowers_obligation_inr'].sum()
                    if x['borrowers_obligation_inr'].sum() > 0 else 0,
        })
    ).reset_index()
    ecl_df['ECL_INR'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['PD']

    st.subheader("Top 15 Projects by ECL")
    st.dataframe(ecl_df.sort_values("ECL_INR", ascending=False).head(15))

# -------------------------------
# Tab 3 – Stress Testing
# -------------------------------
with tab3:
    st.header("Stress Testing")

    beta_gdp = 0.02
    beta_cpi = 0.01
    delta_gdp = st.number_input("GDP Growth Shock (%)", value=-3.0, step=0.5)
    delta_cpi = st.number_input("CPI Inflation Shock (%)", value=2.0, step=0.5)

    ecl_df["pd_stressed"] = ecl_df["PD"] + (beta_gdp * delta_gdp) + (beta_cpi * delta_cpi)
    ecl_df["pd_stressed"] = ecl_df["pd_stressed"].clip(0, 1)

    ecl_df['ECL_stressed'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['pd_stressed']

    st.subheader("Project-level ECL (Baseline vs Stressed)")
    st.dataframe(ecl_df[['project_name', 'ECL_INR', 'ECL_stressed']].sort_values('ECL_stressed', ascending=False).head(15))
