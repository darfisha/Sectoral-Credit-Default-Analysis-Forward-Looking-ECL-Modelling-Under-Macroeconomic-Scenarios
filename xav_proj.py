# -*- coding: utf-8 -*-
"""Streamlit Credit Risk Analysis App.

This app loads and preprocesses the credit risk dataset, trains machine
learning models, and performs a comprehensive financial analysis
including project-level and sectoral ECL calculation and stress testing.
"""

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

# --- Custom CSS for Styling ---
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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #e0e2e6;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 3px solid #3498db;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #3498db;
        color: #3498db;
        background-color: transparent;
    }
    .stButton>button:hover {
        color: #fff;
        background-color: #3498db;
    }
    .css-1d391kg {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions and Data Caching ---

@st.cache_data
def load_and_preprocess_data():
    """Loads and preprocesses the credit risk dataset."""
    file_id = "1MVW1amhh9k3ksDsJkRo9ELvEwRplG0r2"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "credit_risk.csv"

    if not os.path.exists(output):
        with st.spinner("Downloading credit risk data..."):
            gdown.download(url, output, quiet=True)
    
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

    # Filter for India and handle dates/types
    india_df = df[df['country___economy'].str.strip() == 'India'].copy()
    india_df.drop(columns=['currency_of_commitment'], inplace=True, errors='ignore')
    date_cols_india = [
        'end_of_period', 'first_repayment_date', 'last_repayment_date',
        'agreement_signing_date', 'board_approval_date',
        'effective_date_most_recent', 'closed_date_most_recent',
        'last_disbursement_date'
    ]
    for col in date_cols_india:
        if col in india_df.columns:
            india_df[col] = pd.to_datetime(india_df[col], errors='coerce')

    # Add origination year
    india_df["year"] = india_df["agreement_signing_date"].dt.year.astype("Int64")

    # Set financial columns to numeric and handle negative values
    numeric_cols_raw = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd'
    ]
    india_df[numeric_cols_raw] = india_df[numeric_cols_raw].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols_raw:
        india_df[col] = india_df[col].apply(lambda x: np.nan if x < 0 else x)

    # Define 'default_flag' based on loan status and disbursement
    active_statuses = [
        'REPAYING','DISBURSED','DISBURSING','DISBURSING&REPAYING',
        'FULLY DISBURSED','FULLY TRANSFERRED','APPROVED','SIGNED','EFFECTIVE'
    ]
    india_df["is_active"] = india_df["loan_status"].isin(active_statuses).astype(int)
    def encode_default_balanced(status, disbursed_amount):
        if not isinstance(status, str): return 1
        status = status.strip().upper()
        if status in ["FULLY REPAID", "SIGNED", "APPROVED", "DISBURSING"]: return 0
        if status in ["REPAYING", "DISBURSED", "DISBURSING&REPAYING", "FULLY DISBURSED"]: return 1
        if status in ["CANCELLED", "FULLY CANCELLED"]:
            if disbursed_amount and disbursed_amount > 0: return 1
            else: return 0
        return 1
    india_df["default_flag"] = india_df.apply(
        lambda row: encode_default_balanced(row["loan_status"], row["disbursed_amount_ususd"]), axis=1
    )

    # Merge with GDP and CPI data (hardcoded for reproducibility)
    stress_long = pd.DataFrame({
        "year": range(2014, 2025),
        "gdp_growth": [7.4, 8.0, 8.2, 7.0, 6.8, 4.0, -5.8, 9.1, 7.2, 7.3, 6.5],
        "cpi_inflation": [6.7, 5.9, 4.5, 3.6, 3.4, 4.8, 6.6, 5.1, 6.7, 5.7, 5.0]
    })
    merged_df = pd.merge(india_df, stress_long, on="year", how="inner")
    merged_df = merged_df[(merged_df["year"] >= 2014) & (merged_df["year"] <= 2024)].copy()

    # Feature Engineering
    merged_df["repayment_ratio"] = (
        merged_df["repaid_to_ibrd_ususd"] / merged_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    merged_df["loan_to_gdp_growth_ratio"] = (
        merged_df["original_principal_amount_ususd"] / (merged_df["gdp_growth"] * 1e9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return merged_df

@st.cache_resource
def train_models(df):
    """Trains and returns the specified models."""
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
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
        auc = roc_auc_score(y_test, y_prob) if y_test.size > 0 else 0.0
        trained_models[name] = {"model": model, "test_auc": auc, "scaler": scaler, "imputer": imputer}
    return trained_models

@st.cache_resource
def get_all_data_with_predictions(df, models):
    """Predicts default probability for all projects using the best model."""
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd',
        "repayment_ratio", "loan_to_gdp_growth_ratio"
    ]
    
    # Use CatBoost as it's typically the best performer
    model_info = models["CatBoost"]
    catboost_model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]
    
    X_all = df[numeric_cols].values
    X_all_scaled = scaler.transform(imputer.transform(X_all))

    # Predict probability of default (class 1)
    df['default_prob'] = catboost_model.predict_proba(X_all_scaled)[:, 1]
    
    return df

# Define sector classification function
def sector(name: str) -> str:
    n = str(name).upper()
    if any(w in n for w in ["ROAD", "HIGHWAY", "RAIL", "TRANSPORT", "LOGISTICS", "CORRIDOR", "EDFC", "MITP"]): return "Transport & Infrastructure"
    if any(w in n for w in ["POWER", "ENERGY", "SOLAR", "ELECTRIC", "DISTRIBUTION", "24X7"]): return "Energy & Power"
    if any(w in n for w in ["WATER", "IRRIGATION", "DAM", "HYDRO", "BASIN", "WASSIP", "WBADMIP", "KSWMP", "DRIP", "KARN URB WTR", "WTR",  "APIIATP", "SHWSSP" ]): return "Water & Irrigation"
    if any(w in n for w in ["URBAN", "CITY", "HOUSING", "MUNICIPAL", "TOURISM", "TNHHDP", "AMARAVATI", "SWACHH BHARAT"]): return "Urban Development & Housing"
    if any(w in n for w in ["AGRI", "FARM", "RURAL", "LIVELIHOOD", "DAIRY", "FISHERIES", "COOPERATIVE", "JOHAR", "POCRA", "CHIRAAG", "TNRTP", "IAMP"]): return "Agriculture & Rural Development"
    if any(w in n for w in ["HEALTH", "RSHDP", "COVID", "NUTRITION", "DISABILITY", "HSSP", "ICDS", "TB", "SNGRBP", "PHSPP", "EHSDP", "FSPP", "RESPONSIVE SOCIAL PROTECTION"]): return "Health & Social Protection"
    if any(w in n for w in ["EDUCATION", "SCHOOL", "TRAINING", "SKILL", "UNIVERSITY", "WORKFORCE", "NAHEP", "STARS", "GOAL"]): return "Education & Skills"
    if any(w in n for w in ["FINANCE", "MSME", "CREDIT", "BANK", "FISCAL", "PFM", "SAL", "DPL", "PFORR", "RAMP", "BFAIR", "PRIVATE FINANCING"]): return "Finance & Industry"
    if any(w in n for w in ["ISGPP", "CAPABILITY", "GOVERNANCE", "SERVICE DELIVERY", "PSCEP", "SRESTHA", "G-ACRP", "CCP", "UCRRFP", "U-PREPARE"]): return "Governance & Policy Reform"
    if any(w in n for w in ["CLIMATE", "RESILIENT", "LANDSCAPE", "REWARD", "CHALK", "ASSIST", "SMART", "IPSS-CRRA", "AIRBMP", "TRESP"]): return "Environment & Climate"
    if any(w in n for w in ["INNOVATE", "TECH", "ICT", "DIGITAL", "SYSTEMS", "INCLUSION", "ASPIRE", "NECTAR", "INSPIRES", "DAKSH", "KERA"]): return "Technology & Innovation"
    if any(w in n for w in ["DISASTER", "RECOVERY", "REHABILITATION", "RELIEF"]): return "Disaster Recovery & Emergency"
    return "Others"

# --- Main App Logic ---

st.markdown('<h1 class="main-header">Credit Risk Analysis & Prediction 📊</h1>', unsafe_allow_html=True)
st.write("A streamlined, interactive dashboard for assessing credit risk in financial projects.")

# Use a spinner and progress bar for loading
with st.spinner("Loading data and training models..."):
    progress_bar = st.progress(0, text="Loading data...")
    merged_df = load_and_preprocess_data()
    progress_bar.progress(50, text="Training models...")
    trained_models = train_models(merged_df)
    progress_bar.progress(90, text="Predicting default probabilities...")
    merged_df = get_all_data_with_predictions(merged_df, trained_models)
    progress_bar.progress(100, text="App is ready!")
    st.success("Loading complete!")
    st.balloons()
    
# --- Sidebar for User Inputs and Model Selection ---
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
    index=list(trained_models.keys()).index("CatBoost")
)

# --- Tabs for Analysis and Prediction ---
tab1, tab2, tab3 = st.tabs(["Individual Prediction", "Portfolio Analysis", "Stress Testing"])

with tab1:
    st.header("Individual Project Prediction")
    st.info("Adjust the values in the sidebar to predict the probability of default for a single project.")
    
    selected_model_info = trained_models[selected_model_name]
    model = selected_model_info["model"]
    scaler = selected_model_info["scaler"]
    imputer = selected_model_info["imputer"]
    
    # Preprocess the input
    numeric_cols = list(input_features.keys())
    input_df = pd.DataFrame([input_features])
    
    input_df["repayment_ratio"] = (
        input_df["repaid_to_ibrd_ususd"] / input_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    input_df["loan_to_gdp_growth_ratio"] = (
        input_df["original_principal_amount_ususd"] / (input_df["gdp_growth"] * 1e9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    input_features_list = [
        input_df['interest_rate'].iloc[0],
        input_df['original_principal_amount_ususd'].iloc[0],
        input_df['cancelled_amount_ususd'].iloc[0],
        input_df['undisbursed_amount_ususd'].iloc[0],
        input_df['disbursed_amount_ususd'].iloc[0],
        input_df['repaid_to_ibrd_ususd'].iloc[0],
        input_df['due_to_ibrd_ususd'].iloc[0],
        input_df['exchange_adjustment_ususd'].iloc[0],
        input_df['borrowers_obligation_ususd'].iloc[0],
        input_df['loans_held_ususd'].iloc[0],
        input_df['repayment_ratio'].iloc[0],
        input_df['loan_to_gdp_growth_ratio'].iloc[0]
    ]

    input_imputed = imputer.transform([input_features_list])
    input_scaled = scaler.transform(input_imputed)

    # Predict PD
    pd_prob = model.predict_proba(input_scaled)[0, 1]

    # ECL Calculation
    usd_to_inr = 83
    LGD = 0.45 # A simple LGD assumption based on your notebook
    EAD_INR = input_features['borrowers_obligation_ususd'] * usd_to_inr
    ECL_project = EAD_INR * LGD * pd_prob

    st.subheader(f"Results using {selected_model_name}")
    st.metric("Probability of Default (PD)", f"{pd_prob:.2%}")
    st.metric("Expected Credit Loss (ECL)", f"₹ {ECL_project:,.2f}")
    
with tab2:
    st.header("Overall Portfolio & Sectoral Analysis")

    # Project-level ECL
    st.subheader("Top Projects by ECL")
    usd_to_inr = 83
    merged_df['borrowers_obligation_inr'] = merged_df['borrowers_obligation_ususd'] * usd_to_inr
    merged_df['repaid_to_ibrd_inr'] = merged_df['repaid_to_ibrd_ususd'] * usd_to_inr
    merged_df['LGD'] = ((merged_df['borrowers_obligation_inr'] - merged_df['repaid_to_ibrd_inr']) / merged_df['borrowers_obligation_inr'])
    merged_df.loc[merged_df['default_flag'] == 0, 'LGD'] = 0
    merged_df["sector"] = merged_df["project_name"].apply(sector)

    ecl_df = merged_df.groupby('project_name').apply(
        lambda x: pd.Series({
            'EAD_INR': x['borrowers_obligation_inr'].sum(),
            'PD': x['default_prob'].mean(),
            'LGD': ((x['borrowers_obligation_inr'] * x['LGD']).sum()) / x['borrowers_obligation_inr'].sum() if x['borrowers_obligation_inr'].sum() > 0 else 0,
            'sector': x['sector'].iloc[0]
        })
    ).reset_index()
    ecl_df['ECL_INR'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['PD']
    st.dataframe(ecl_df[['project_name', 'ECL_INR', 'PD', 'LGD', 'sector']].sort_values('ECL_INR', ascending=False).head(20))
    
    # Sectoral ECL
    st.subheader("Sector-wise Expected Credit Loss (ECL)")
    sector_ecl = ecl_df.groupby('sector').agg(
        total_projects=('project_name', 'nunique'),
        total_ECL_INR=('ECL_INR', 'sum')
    ).reset_index()
    sector_ecl['total_ECL_BN'] = sector_ecl['total_ECL_INR'] / 1e9
    sector_ecl = sector_ecl.sort_values('total_ECL_BN', ascending=False)
    st.dataframe(sector_ecl[['sector', 'total_projects', 'total_ECL_BN']])

    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=sector_ecl,
            x='total_ECL_BN',
            y='sector',
            palette='viridis',
            ax=ax
        )
        ax.set_xlabel('Total ECL (Billion INR)')
        ax.set_ylabel('Sector')
        ax.set_title('Sector-wise Expected Credit Loss')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_projects_plot = ecl_df.sort_values("ECL_INR", ascending=False).head(15)
        ax.barh(top_projects_plot["project_name"], top_projects_plot["ECL_INR"]/1e9, color="tomato")
        ax.invert_yaxis()
        ax.set_title("Top 15 Projects by ECL", fontsize=14)
        ax.set_xlabel("ECL (INR Billion)")
        ax.set_ylabel("Project")
        st.pyplot(fig)

with tab3:
    st.header("Stress Testing Analysis")
    st.write("This analysis simulates the impact of an economic downturn on the portfolio.")
    
    # Stress scenario parameters
    beta_gdp = 0.02
    beta_cpi = 0.01
    delta_gdp = st.number_input("GDP Growth Shock (%)", value=-3.0, step=0.5)
    delta_cpi = st.number_input("CPI Inflation Shock (%)", value=2.0, step=0.5)

    # Calculate stressed PD
    ecl_df["pd_stressed"] = ecl_df["PD"] + (beta_gdp * delta_gdp) + (beta_cpi * delta_cpi)
    ecl_df["pd_stressed"] = ecl_df["pd_stressed"].clip(0,1)

    # Calculate stressed ECL
    ecl_df['ECL_stressed'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['pd_stressed']
    ecl_df['ECL_stressed_BN'] = ecl_df['ECL_stressed'] / 1e9

    st.subheader("Project-level ECL (Baseline vs Stressed)")
    project_ecl_comparison = ecl_df[['project_name', 'ECL_INR', 'ECL_stressed']].sort_values('ECL_stressed', ascending=False)
    st.dataframe(project_ecl_comparison.head(20))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_projects_stress = ecl_df.sort_values('ECL_stressed', ascending=False).head(15)
    bar_width = 0.4
    indices = np.arange(len(top_projects_stress))
    ax.barh(indices - bar_width/2, top_projects_stress['ECL_INR']/1e9, bar_width, label='Baseline ECL', color='skyblue')
    ax.barh(indices + bar_width/2, top_projects_stress['ECL_stressed']/1e9, bar_width, label='Stressed ECL', color='salmon')
    ax.set_yticks(indices, top_projects_stress['project_name'])
    ax.set_xlabel('ECL (INR Billion)')
    ax.set_ylabel('Project')
    ax.set_title('Top 15 Projects: Baseline vs Stressed ECL')
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("Sector-level ECL (Baseline vs Stressed)")
    sector_ecl_stress = ecl_df.groupby('sector').agg(
        ECL_baseline_BN=('ECL_INR', 'sum'),
        ECL_stressed_BN=('ECL_stressed', 'sum')
    ).reset_index()
    sector_ecl_stress['ECL_baseline_BN'] /= 1e9
    sector_ecl_stress['ECL_stressed_BN'] /= 1e9
    sector_ecl_stress = sector_ecl_stress.sort_values('ECL_stressed_BN', ascending=False)
    st.dataframe(sector_ecl_stress)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_plot = sector_ecl_stress.sort_values('ECL_stressed_BN', ascending=False)
    indices = np.arange(len(sector_plot))
    bar_width = 0.4
    ax.barh(indices - bar_width/2, sector_plot['ECL_baseline_BN'], bar_width, label='Baseline ECL', color='skyblue')
    ax.barh(indices + bar_width/2, sector_plot['ECL_stressed_BN'], bar_width, label='Stressed ECL', color='salmon')
    ax.set_yticks(indices, sector_plot['sector'])
    ax.set_xlabel('Total ECL (INR Billion)')
    ax.set_ylabel('Sector')
    ax.set_title('Sector-level ECL: Baseline vs Stressed')
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)
