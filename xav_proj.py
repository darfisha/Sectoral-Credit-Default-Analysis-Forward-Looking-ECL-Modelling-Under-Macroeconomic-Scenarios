import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import gdown

# -----------------------------
# 1️⃣ Load Data with Caching
# -----------------------------
@st.cache_data
def load_and_preprocess_data():
    """Loads and preprocesses the credit risk dataset from the local file."""
    file_path = "merged_df.csv"
    if not os.path.exists(file_path):
        st.error("The 'merged_df.csv' file was not found. Please make sure it is in the same directory as the app.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        st.error(f"Error loading data from file: {e}")
        return pd.DataFrame()

    # Robustly check for and rename project name column
    if 'project_name' in df.columns:
        df.rename(columns={'project_name': 'ProjectName'}, inplace=True)
    elif 'ProjectName' not in df.columns:
        st.error("The dataset must contain a 'project_name' or 'ProjectName' column.")
        return pd.DataFrame()
    
    # Fill missing ProjectName values
    df['ProjectName'] = df['ProjectName'].fillna("Unknown Project")

    # Clean other columns if necessary (matching previous logic)
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("$", "usd").replace("'", "").replace(".", "")
        for col in df.columns
    ]

    # --- Dynamic sector assignment (based on project attributes) ---
    def assign_sector(name: str) -> str:
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

    df['Sector'] = df['ProjectName'].apply(assign_sector)
    
    return df

# -----------------------------
# 2️⃣ Train models once (cached)
# -----------------------------
@st.cache_resource
def train_models(df):
    features = df.select_dtypes(include='number').columns.tolist()
    
    # Filter out target variables to avoid data leakage
    features_to_predict = ['default_flag', 'default_prob', 'ECL']
    features = [f for f in features if f not in features_to_predict]
    
    # Handle case where default_flag column is missing
    if 'default_flag' not in df.columns:
        st.error("Missing 'default_flag' column in the dataset. Cannot train models.")
        return {}, []

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

# -----------------------------
# 3️⃣ Prediction function
# -----------------------------
@st.cache_data
def predict_df(df, model_name, features):
    model_info = models[model_name]
    model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    X = df[features].values
    X_scaled = scaler.transform(imputer.transform(X))

    df = df.copy()
    df['default_prob'] = model.predict(X_scaled).clip(0, 1) # Ensure probabilities are between 0 and 1
    
    # The user's provided logic for ECL is simpler here:
    df['ECL'] = df['default_prob'] * df.get('original_principal_amount_ususd', 1)

    sectoral_pd = df.groupby('Sector')['default_prob'].mean().reset_index()
    sectoral_ecl = df.groupby('Sector')['ECL'].sum().reset_index()

    return df, sectoral_pd, sectoral_ecl

# -----------------------------
# 4️⃣ Streamlit Layout
# -----------------------------
st.title("Credit Risk Analysis & ECL Modelling 📈")

merged_df = load_and_preprocess_data()
if merged_df.empty:
    st.stop()
    
models, features = train_models(merged_df)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Model Selection", "Project-Level Prediction", "Sectoral Analysis", "Stress Testing"])

# --- Homepage ---
if page == "Homepage":
    st.header("App Overview & Data Preview")
    st.write("This app analyzes credit risk for projects, predicts default probabilities, and simulates financial stress scenarios.")
    st.subheader("Dataset Preview")
    st.write("Below are the first 5 rows of the dataset used for this analysis.")
    st.dataframe(merged_df.head())
    st.write(f"Dataset Shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

# --- Model Selection ---
elif page == "Model Selection":
    st.header("Select Regression Model")
    model_name = st.selectbox("Choose a model for your analysis:", ["CatBoost", "XGBoost", "RandomForest"])
    st.write(f"Using model: **{model_name}**")
    st.write("Each model has been pre-trained to predict the probability of default based on financial features.")
    
# --- Project-Level Prediction ---
elif page == "Project-Level Prediction":
    st.header("Project-Level Prediction & Analysis")
    model_name = st.selectbox("Select a model to use for prediction:", ["CatBoost", "XGBoost", "RandomForest"])
    preds_df, _, _ = predict_df(merged_df, model_name, features)
    st.subheader("Project-level PD & ECL")
    st.write("The table below shows the top 10 projects by their predicted Probability of Default (PD) and Expected Credit Loss (ECL).")
    st.dataframe(preds_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].sort_values(by='ECL', ascending=False).head(10))

# --- Sectoral Analysis ---
elif page == "Sectoral Analysis":
    st.header("Sectoral PD & ECL Analysis")
    model_name = st.selectbox("Select a model to use for prediction:", ["CatBoost", "XGBoost", "RandomForest"])
    _, sectoral_pd, sectoral_ecl = predict_df(merged_df, model_name, features)
    st.subheader("Average Sectoral PD")
    st.write("This table shows the average probability of default across different sectors.")
    st.dataframe(sectoral_pd.sort_values(by='default_prob', ascending=False))
    st.subheader("Total Sectoral ECL")
    st.write("This table shows the total expected credit loss for each sector.")
    st.dataframe(sectoral_ecl.sort_values(by='ECL', ascending=False))

# --- Stress Testing ---
elif page == "Stress Testing":
    st.header("Stress Testing Scenario Simulation")
    model_name = st.selectbox("Select a model to use for simulation:", ["CatBoost", "XGBoost", "RandomForest"])
    st.write("Define a stress scenario by adjusting the percentage change for key features below. The app will then predict the new ECL.")
    
    # Define features for stress scenario
    stress_features = ['interest_rate', 'original_principal_amount_ususd', 'repaid_to_ibrd_ususd', 'due_to_ibrd_ususd']
    scenario_changes = {}
    for feature in stress_features:
        scenario_changes[feature] = st.slider(f"{feature} change (%)", -50, 100, 0)

    if st.button("Apply Stress Scenario"):
        stressed_df = merged_df.copy()
        for feature, change in scenario_changes.items():
            if feature in stressed_df.columns:
                stressed_df[feature] = stressed_df[feature] * (1 + change / 100)

        stressed_df, sectoral_pd, sectoral_ecl = predict_df(stressed_df, model_name, features)
        
        st.subheader("Sectoral PD under stress scenario")
        st.dataframe(sectoral_pd.sort_values(by='default_prob', ascending=False))
        st.subheader("Sectoral ECL under stress scenario")
        st.dataframe(sectoral_ecl.sort_values(by='ECL', ascending=False))
        st.write("Project-level PD & ECL (first 10 rows) under the new scenario:")
        st.dataframe(stressed_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].head(10))
