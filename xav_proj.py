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

@st.cache_data
def predict_single(input_df, model_name, features):
    model_info = models[model_name]
    model = model_info['model']
    scaler = model_info['scaler']
    imputer = model_info['imputer']

    X = input_df[features].values
    X_scaled = scaler.transform(imputer.transform(X))

    input_df['default_prob'] = model.predict(X_scaled).clip(0,1)
    principal_col = 'original_principal_amount_ususd'
    input_df['ecl'] = input_df['default_prob'] * input_df.get(principal_col, 1)

    return input_df[['sector', 'default_prob', 'ecl']]

# -----------------------------
# 2️⃣ App Layout
# -----------------------------
st.title("Credit Risk Prediction 💰")

merged_df = load_and_preprocess_data()
if merged_df.empty:
    st.stop()

models, features = train_models(merged_df)

st.header("Enter Project Details for Prediction")

# Select sector
sector_options = merged_df['sector'].unique().tolist()
selected_sector = st.selectbox("Select Project Sector", sector_options)

# Select model
model_name = st.selectbox("Select Model", ["CatBoost", "XGBoost", "RandomForest"])

# User inputs for top numeric features (top 4)
numeric_cols = [f for f in merged_df.select_dtypes(include='number').columns if f not in ['default_flag', 'default_prob', 'ecl']]
input_features = numeric_cols[:4]  # top 4 numeric features for simplicity

user_input = {}
for f in input_features:
    user_input[f] = st.number_input(f"Enter {f}", value=float(merged_df[f].median()))

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])
input_df['sector'] = selected_sector  # add sector column

# Predict
if st.button("Predict PD & ECL"):
    result = predict_single(input_df, model_name, features)
    st.subheader("Prediction Result")
    st.write(result)
