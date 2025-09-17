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

# Safe SHAP import for Python 3.13
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
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

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

    X_train, y_train, X_val, y_val, X_test, y_test = split_chronologically(df, numeric_cols)

    st.subheader("Model Training & Evaluation")
    results_df = train_evaluate_models(X_train, y_train, X_test, y_test)
    st.dataframe(results_df)
