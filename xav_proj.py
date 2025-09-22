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

# --- Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Data Loading & Preprocessing
# ===============================
@st.cache_data
def load_and_preprocess_data():
    file_id = "1MVW1amhh9k3ksDsJkRo9ELvEwRplG0r2"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "credit_risk.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    date_cols = [
        "Agreement Signing Date", "Board Approval Date", "Closed Date (Most Recent)",
        "Effective Date (Most Recent)", "First Repayment Date",
        "Last Disbursement Date", "Last Repayment Date",
    ]
    df = pd.read_csv(output, parse_dates=date_cols, low_memory=False)

    # Clean column names
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("$", "usd").replace("'", "").replace(".", "")
        for col in df.columns
    ]

    # Filter for India
    india_df = df[df['country___economy'].str.strip() == 'India'].copy()
    india_df.drop(columns=['currency_of_commitment'], inplace=True, errors='ignore')

    # Date conversions
    date_cols_india = [
        'end_of_period', 'first_repayment_date', 'last_repayment_date',
        'agreement_signing_date', 'board_approval_date',
        'effective_date_most_recent', 'closed_date_most_recent',
        'last_disbursement_date'
    ]
    for col in date_cols_india:
        if col in india_df.columns:
            india_df[col] = pd.to_datetime(india_df[col], errors='coerce')

    india_df["year"] = india_df["agreement_signing_date"].dt.year.astype("Int64")

    # Financial columns
    numeric_cols_raw = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd'
    ]
    india_df[numeric_cols_raw] = india_df[numeric_cols_raw].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols_raw:
        india_df[col] = india_df[col].apply(lambda x: np.nan if x < 0 else x)

    # Default flag
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

    # GDP + CPI mock data
    stress_long = pd.DataFrame({
        "year": range(2014, 2025),
        "gdp_growth": [7.4, 8.0, 8.2, 7.0, 6.8, 4.0, -5.8, 9.1, 7.2, 7.3, 6.5],
        "cpi_inflation": [6.7, 5.9, 4.5, 3.6, 3.4, 4.8, 6.6, 5.1, 6.7, 5.7, 5.0]
    })
    merged_df = pd.merge(india_df, stress_long, on="year", how="inner")
    merged_df = merged_df[(merged_df["year"] >= 2014) & (merged_df["year"] <= 2024)].copy()

    # Feature engineering
    merged_df["repayment_ratio"] = (
        merged_df["repaid_to_ibrd_ususd"] / merged_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    merged_df["loan_to_gdp_growth_ratio"] = (
        merged_df["original_principal_amount_ususd"] / (merged_df["gdp_growth"] * 1e9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return merged_df

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

    train_df = df[(df["year"] >= 2014) & (df["year"] <= 2020)].copy()
    test_df  = df[(df["year"] >= 2023) & (df["year"] <= 2024)].copy()

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
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if y_test.size > 0 else [0]
        auc = roc_auc_score(y_test, y_prob) if y_test.size > 0 else 0.0
        trained_models[name] = {"model": model, "test_auc": auc, "scaler": scaler, "imputer": imputer}
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

st.write("✅ App is ready! Use sidebar for inputs and explore results.")

