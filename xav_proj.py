# ===============================
# xav_proj_streamlit.py
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Safe SHAP import
try:
    import shap
except ImportError:
    shap = None
    st.warning("SHAP not available. Skipping SHAP interpretability.")

# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_credit_data(file_path: str, india_only=True):
    df = pd.read_csv(file_path, parse_dates=True, low_memory=False)
    df.columns = [col.strip().lower().replace(" ", "_").replace("/", "_")
                  .replace("(", "").replace(")", "").replace("$", "usd")
                  .replace("'", "").replace(".", "") for col in df.columns]
    if india_only and 'country___economy' in df.columns:
        df = df[df['country___economy'].str.strip() == 'India'].copy()
    date_cols = ['end_of_period','first_repayment_date','last_repayment_date',
                 'agreement_signing_date','board_approval_date','effective_date_most_recent',
                 'closed_date_most_recent','last_disbursement_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'agreement_signing_date' in df.columns:
        df["origination_year"] = df["agreement_signing_date"].dt.year.astype("Int64")
    return df

# ===============================
# FEATURE ENGINEERING
# ===============================
def preprocess_financials(df):
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd', 'borrowers_obligation_ususd', 'loans_held_ususd'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    return df, numeric_cols

def encode_default_flags(df):
    active_statuses = ['REPAYING','DISBURSED','DISBURSING','DISBURSING&REPAYING',
                       'FULLY DISBURSED','FULLY TRANSFERRED','APPROVED','SIGNED','EFFECTIVE']
    df['loan_status'] = df['loan_status'].astype(str).str.strip().str.upper()
    df["is_active"] = df["loan_status"].isin(active_statuses).astype(int)
    def encode_default_balanced(status, disbursed_amount):
        if not isinstance(status, str):
            return 1
        status = status.strip().upper()
        if status in ["FULLY REPAID", "SIGNED", "APPROVED", "DISBURSING"]:
            return 0
        if status in ["REPAYING", "DISBURSED", "DISBURSING&REPAYING", "FULLY DISBURSED"]:
            return 1
        if status in ["CANCELLED", "FULLY CANCELLED"]:
            return 1 if disbursed_amount and disbursed_amount > 0 else 0
        return 1
    df["default_flag"] = df.apply(lambda row: encode_default_balanced(row["loan_status"], row["disbursed_amount_ususd"]), axis=1)
    return df

def add_engineered_features(df):
    if 'agreement_signing_date' in df.columns:
        df['year'] = df['agreement_signing_date'].dt.year
    if 'gdp_growth' in df.columns:
        df['loan_to_gdp_growth_ratio'] = (df['original_principal_amount_ususd'] / (df['gdp_growth']*1e9)).replace([np.inf, -np.inf], 0)
    df['repayment_ratio'] = (df['repaid_to_ibrd_ususd'] / df['disbursed_amount_ususd']).replace([np.inf, -np.inf], 0)
    return df

# ===============================
# TRAIN/VAL/TEST SPLIT
# ===============================
def split_chronologically(df, numeric_cols):
    train_df = df[(df["year"] >= 2014) & (df["year"] <= 2020)].copy()
    val_df   = df[(df["year"] >= 2021) & (df["year"] <= 2022)].copy()
    test_df  = df[(df["year"] >= 2023) & (df["year"] <= 2024)].copy()
    def get_Xy(df_sub):
        X = df_sub[numeric_cols].values
        y = df_sub["default_flag"].values
        return X, y
    X_train, y_train = get_Xy(train_df)
    X_val, y_val     = get_Xy(val_df)
    X_test, y_test   = get_Xy(test_df)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_scaled   = scaler.transform(imputer.transform(X_val))
    X_test_scaled  = scaler.transform(imputer.transform(X_test))
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, train_df, val_df, test_df

# ===============================
# MODEL TRAINING
# ===============================
def train_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "Naive Bayes": GaussianNB()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
        results.append([name,
                        accuracy_score(y_test, y_pred),
                        precision_score(y_test, y_pred, zero_division=0),
                        recall_score(y_test, y_pred, zero_division=0),
                        f1_score(y_test, y_pred, zero_division=0),
                        roc_auc_score(y_test, y_prob)])
    results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC"])
    results_df = results_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df

# ===============================
# ECL CALCULATION
# ===============================
def compute_ecl(merged_df, usd_to_inr=83):
    merged_df = merged_df.sort_values(['project_name','end_of_period']).dropna(subset=['borrowers_obligation_ususd'])
    latest_dates = merged_df.groupby('project_name')['end_of_period'].max().reset_index()
    latest_loans = pd.merge(merged_df, latest_dates, on=['project_name','end_of_period'], how='inner')
    pd_weighted = latest_loans.groupby('project_name').apply(
        lambda x: pd.Series({'PD': (x['borrowers_obligation_ususd']*usd_to_inr*x['default_flag']).sum() / (x['borrowers_obligation_ususd']*usd_to_inr).sum()})
    ).reset_index()
    ead_summary = latest_loans.groupby('project_name', as_index=False).agg(
        total_active_loans=('loan_number','count'),
        EAD_INR=('borrowers_obligation_ususd', lambda x: x.sum()*usd_to_inr)
    )
    merged_df['borrowers_obligation_inr'] = merged_df['borrowers_obligation_ususd'] * usd_to_inr
    merged_df['LGD'] = merged_df['borrowers_obligation_inr'] * merged_df['default_flag']
    ecl_df = pd.merge(ead_summary, pd_weighted, on='project_name', how='left')
    ecl_df['LGD'] = merged_df.groupby('project_name')['LGD'].sum().values
    ecl_df['ECL'] = ecl_df['PD'] * ecl_df['LGD']
    return ecl_df

# ===============================
# VISUALIZATION
# ===============================
def plot_top_projects(ecl_df, top_n=10):
    top_projects = ecl_df.sort_values('ECL', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='project_name', y='ECL', data=top_projects, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("Top Projects by ECL")
    st.pyplot(fig)

# ===============================
# STREAMLIT APP
# ===============================
st.title("Sectoral Credit Default Analysis (ECL Model)")

uploaded_file = st.file_uploader("Upload credit data CSV", type="csv")

if uploaded_file is not None:
    df = load_credit_data(uploaded_file)
    st.write("Raw Data Sample:")
    st.dataframe(df.head())

    df = encode_default_flags(df)
    df, numeric_cols = preprocess_financials(df)
    df = add_engineered_features(df)

    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = split_chronologically(df, numeric_cols)

    st.subheader("Model Training & Evaluation")
    results_df = train_evaluate_models(X_train, y_train, X_test, y_test)
    st.dataframe(results_df)

    st.subheader("Expected Credit Loss (ECL) Calculation")
    ecl_df = compute_ecl(df)
    st.dataframe(ecl_df.sort_values("ECL", ascending=False).head(10))

    st.subheader("Top Projects by ECL")
    top_n = st.slider("Number of top projects to show", min_value=5, max_value=20, value=10)
    plot_top_projects(ecl_df, top_n=top_n)
