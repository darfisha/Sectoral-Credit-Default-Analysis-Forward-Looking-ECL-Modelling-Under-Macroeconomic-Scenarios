import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os

# -----------------------------
# 1️⃣ Load Data & Train Models
# -----------------------------
@st.cache_data
def load_and_preprocess_data():
    file_path = "merged_df.csv"
    if not os.path.exists(file_path):
        st.error("The 'merged_df.csv' file was not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, low_memory=False)

    if 'project_name' in df.columns:
        df.rename(columns={'project_name': 'ProjectName'}, inplace=True)
    df['ProjectName'] = df['ProjectName'].fillna("Unknown Project")

    # Assign Sector
    def assign_sector(name: str) -> str:
        n = str(name).upper()
        if any(w in n for w in ["ROAD", "HIGHWAY", "RAIL", "TRANSPORT", "LOGISTICS", "CORRIDOR", "EDFC", "MITP"]): return "Transport & Infrastructure"
        if any(w in n for w in ["POWER", "ENERGY", "SOLAR", "ELECTRIC", "DISTRIBUTION", "24X7"]): return "Energy & Power"
        if any(w in n for w in ["WATER", "IRRIGATION", "DAM", "HYDRO", "BASIN", "WASSIP", "WBADMIP", "KSWMP", "DRIP", "KARN URB WTR", "WTR", "APIIATP", "SHWSSP"]): return "Water & Irrigation"
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

    df['sector'] = df['ProjectName'].apply(assign_sector)

    # Clean numeric columns
    cols_to_clean = [col for col in df.columns if col not in ['ProjectName', 'sector']]
    cleaned_cols = {col: col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("$", "usd").replace("'", "").replace(".", "") for col in cols_to_clean}
    df.rename(columns=cleaned_cols, inplace=True)

    return df

@st.cache_resource
def train_models(df):
    features_to_predict = ['default_flag', 'default_prob', 'ecl']
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    features = [col for col in numeric_cols if col not in features_to_predict]

    X = df[features].values
    y = df['default_flag'].values

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    cat_model = CatBoostRegressor(iterations=200, verbose=0)
    cat_model.fit(X_scaled, y)

    xgb_model = XGBRegressor(n_estimators=200, eval_metric='rmse', random_state=42)
    xgb_model.fit(X_scaled, y)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_scaled, y)

    models = {
        "CatBoost": {"model": cat_model, "scaler": scaler, "imputer": imputer},
        "XGBoost": {"model": xgb_model, "scaler": scaler, "imputer": imputer},
        "RandomForest": {"model": rf_model, "scaler": scaler, "imputer": imputer}
    }
    return models, features

# Prediction function
@st.cache_data
def predict_df(input_df, model_name, features):
    model_info = models[model_name]
    model = model_info['model']
    scaler = model_info['scaler']
    imputer = model_info['imputer']

    # Ensure all features exist
    full_input = pd.DataFrame(columns=features)
    for col in features:
        if col in input_df.columns:
            full_input[col] = input_df[col]
        else:
            full_input[col] = merged_df[col].median()

    X_scaled = scaler.transform(imputer.transform(full_input.values))
    input_df['default_prob'] = model.predict(X_scaled).clip(0,1)
    input_df['ecl'] = input_df['default_prob'] * input_df.get('original_principal_amount_ususd', 1)

    return input_df

# -----------------------------
# 2️⃣ App Layout
# -----------------------------
st.title("Credit Risk Analysis & ECL Modelling 📈")

merged_df = load_and_preprocess_data()
if merged_df.empty:
    st.stop()

models, features = train_models(merged_df)

page = st.sidebar.radio("Navigation", ["Homepage", "Project Prediction", "Stress Testing"])

# --- Homepage ---
if page == "Homepage":
    st.header("App Overview")
    st.write("""
    Predict **Probability of Default (PD)** and **Expected Credit Loss (ECL)** for projects.
    Enter project details for individual prediction or simulate stress scenarios like CPI and GDP changes.
    """)

# --- Project Prediction ---
elif page == "Project Prediction":
    st.header("Project-Level Prediction")

    sector_options = merged_df['sector'].unique().tolist()
    selected_sector = st.selectbox("Select Project Sector", sector_options)

    model_name = st.selectbox("Select Model", ["CatBoost", "XGBoost", "RandomForest"])

    interest_rate = st.number_input("Enter interest_rate", value=float(merged_df['interest_rate'].median()))
    principal = st.number_input("Enter original_principal_amount_ususd", value=float(merged_df['original_principal_amount_ususd'].median()))

    input_df = pd.DataFrame([{
        'interest_rate': interest_rate,
        'original_principal_amount_ususd': principal,
        'sector': selected_sector
    }])

    if st.button("Predict PD & ECL"):
        result = predict_df(input_df, model_name, features)
        st.subheader("Prediction Result")
        st.write(result[['sector','default_prob','ecl']])

# --- Stress Testing ---
elif page == "Stress Testing":
    st.header("Stress Testing Simulation")

    model_name = st.selectbox("Select Model", ["CatBoost", "XGBoost", "RandomForest"], key="stress_model")

    cpi_change = st.slider("CPI change (%)", -50, 100, 0)
    gdp_change = st.slider("GDP change (%)", -50, 100, 0)

    stressed_df = merged_df.copy()
    stressed_df['interest_rate'] = stressed_df['interest_rate'] * (1 + cpi_change / 100)
    stressed_df['original_principal_amount_ususd'] = stressed_df['original_principal_amount_ususd'] * (1 + gdp_change / 100)

    stressed_df = predict_df(stressed_df, model_name, features)
    st.subheader("PD & ECL under Stress Scenario")
    st.write(stressed_df[['ProjectName' if 'ProjectName' in stressed_df.columns else 'sector','default_prob','ecl']])
