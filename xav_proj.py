# -*- coding: utf-8 -*-
"""Credit Risk Analysis Script.

This script loads the credit risk dataset, preprocesses the data,
trains machine learning models, and performs a comprehensive financial analysis
including project-level and sectoral ECL calculation and stress testing.

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import gdown

# --- 1. Data Loading and Preprocessing ---
print("--- Step 1: Loading and Preprocessing Data ---")

file_id = "1MVW1amhh9k3ksDsJkRo9ELvEwRplG0r2"
url = f"https://drive.google.com/uc?id={file_id}"
output = "credit_risk.csv"

if not os.path.exists(output):
    print("Downloading file...")
    gdown.download(url, output, quiet=False)
else:
    print(f"File '{output}' already exists. Skipping download.")

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

# Final numeric features list
numeric_cols = [
    'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
    'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
    'due_to_ibrd_ususd','exchange_adjustment_ususd',
    'borrowers_obligation_ususd', 'loans_held_ususd',
    "repayment_ratio", "loan_to_gdp_growth_ratio"
]

# Data Split for training
train_df = merged_df[(merged_df["year"] >= 2014) & (merged_df["year"] <= 2020)].copy()
val_df   = merged_df[(merged_df["year"] >= 2021) & (merged_df["year"] <= 2022)].copy()
test_df  = merged_df[(merged_df["year"] >= 2023) & (merged_df["year"] <= 2024)].copy()

X_train = train_df[numeric_cols].values
y_train = train_df["default_flag"].values
X_test  = test_df[numeric_cols].values if not test_df.empty else np.empty((0, len(numeric_cols)))
y_test  = test_df["default_flag"].values if not test_df.empty else np.array([])

# Impute and Scale
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed  = imputer.transform(X_test) if X_test.size else X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled  = scaler.transform(X_test_imputed) if X_test_imputed.size else X_test_imputed

print("Preprocessing complete. Training models.")

# --- 2. Train Models and Predict Probability of Default (PD) ---
print("\n--- Step 2: Training Models for PD Prediction ---")

models = {
    "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"Model": model, "Test AUC": auc}

# Use the best model (CatBoost from the notebook) for the analysis
best_model = results["CatBoost"]["Model"]
X_all = merged_df[numeric_cols].values
X_all_scaled = scaler.transform(imputer.transform(X_all))
merged_df['default_prob'] = best_model.predict_proba(X_all_scaled)[:, 1]

# --- 3. Financial Analysis and ECL Calculation ---
print("\n--- Step 3: Performing Financial Analysis and ECL Calculation ---")

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

merged_df["sector"] = merged_df["project_name"].apply(sector)

# Project-level ECL
print("\n--- Project-Level ECL Calculation ---")
usd_to_inr = 83
merged_df['borrowers_obligation_inr'] = merged_df['borrowers_obligation_ususd'] * usd_to_inr
merged_df['repaid_to_ibrd_inr'] = merged_df['repaid_to_ibrd_ususd'] * usd_to_inr

# Calculate LGD for all loans
merged_df['LGD'] = ((merged_df['borrowers_obligation_inr'] - merged_df['repaid_to_ibrd_inr']) / merged_df['borrowers_obligation_inr'])
merged_df.loc[merged_df['default_flag'] == 0, 'LGD'] = 0

# Aggregate PD, EAD, LGD per project
ecl_df = merged_df.groupby('project_name').apply(
    lambda x: pd.Series({
        'EAD_INR': x['borrowers_obligation_inr'].sum(),
        'PD': x['default_prob'].mean(), # Use the predicted PD
        'LGD': ((x['borrowers_obligation_inr'] * x['LGD']).sum()) / x['borrowers_obligation_inr'].sum() if x['borrowers_obligation_inr'].sum() > 0 else 0
    })
).reset_index()
ecl_df['ECL_INR'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['PD']
print("Top 20 Projects by Expected Credit Loss (ECL):")
print(ecl_df.sort_values('ECL_INR', ascending=False).head(20))

# Sectoral ECL and Analysis
print("\n--- Sectoral Analysis ---")
ecl_df = ecl_df.merge(merged_df[['project_name', 'sector']].drop_duplicates(), on='project_name', how='left')
sector_ecl = ecl_df.groupby('sector').agg(
    total_projects=('project_name', 'nunique'),
    total_ECL_INR=('ECL_INR', 'sum')
).reset_index()
sector_ecl['total_ECL_BN'] = sector_ecl['total_ECL_INR'] / 1e9
sector_ecl = sector_ecl.sort_values('total_ECL_BN', ascending=False)
print("Sector-wise Expected Credit Loss (ECL):")
print(sector_ecl[['sector', 'total_projects', 'total_ECL_BN']])

# ECL Under Stress Testing
print("\n--- ECL Under Stress Testing ---")
# Define stress scenario parameters (hardcoded from notebook)
beta_gdp = 0.02
beta_cpi = 0.01
delta_gdp = -3.0
delta_cpi = 2.0

# Calculate stressed PD
ecl_df["pd_stressed"] = ecl_df["PD"] + (beta_gdp * delta_gdp) + (beta_cpi * delta_cpi)
ecl_df["pd_stressed"] = ecl_df["pd_stressed"].clip(0,1)

# Calculate stressed ECL
ecl_df['ECL_stressed'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['pd_stressed']
ecl_df['ECL_stressed_BN'] = ecl_df['ECL_stressed'] / 1e9

project_ecl_stress = ecl_df[['project_name', 'ECL_stressed_BN']].sort_values('ECL_stressed_BN', ascending=False)
print("Top 20 Projects by Stressed ECL:")
print(project_ecl_stress.head(20))

# Compare Baseline vs Stressed ECL at Project and Sector Level
ecl_df['ECL_baseline_BN'] = ecl_df['ECL_INR'] / 1e9
project_ecl_comparison = ecl_df[['project_name', 'ECL_baseline_BN', 'ECL_stressed_BN']].sort_values('ECL_stressed_BN', ascending=False)
print("\nProject-level ECL Comparison (Baseline vs Stressed):")
print(project_ecl_comparison.head(10))

sector_ecl_stress = ecl_df.groupby('sector').agg(
    ECL_baseline_BN=('ECL_baseline_BN', 'sum'),
    ECL_stressed_BN=('ECL_stressed_BN', 'sum')
).reset_index().sort_values('ECL_stressed_BN', ascending=False)
print("\nSector-level ECL Comparison (Baseline vs Stressed):")
print(sector_ecl_stress)
