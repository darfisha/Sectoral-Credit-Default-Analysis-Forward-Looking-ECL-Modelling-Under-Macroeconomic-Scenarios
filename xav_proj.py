import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1️⃣ Load CSV from GitHub
# -----------------------------
RAW_URL = "https://raw.githubusercontent.com/darfisha/Sectoral-Credit-Default-Analysis-Forward-Looking-ECL-Modelling-Under-Macroeconomic-Scenarios/main/merged_df.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

merged_df = load_data(RAW_URL)

# Ensure ProjectName and Sector exist
if 'ProjectName' not in merged_df.columns:
    merged_df['ProjectName'] = ["Project_" + str(i) for i in range(len(merged_df))]
if 'Sector' not in merged_df.columns:
    merged_df['Sector'] = ["Sector_" + str(i % 3 + 1) for i in range(len(merged_df))]

# -----------------------------
# 2️⃣ Train models once (cached)
# -----------------------------
@st.cache_resource
def train_models(df):
    features = df.select_dtypes(include='number').columns.drop('default_flag').tolist()
    X = df[features].values
    y = df['default_flag'].values

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    cat_model = CatBoostRegressor(iterations=200, verbose=0)
    cat_model.fit(X_scaled, y)

    xgb_model = XGBRegressor(n_estimators=200, eval_metric='rmse')
    xgb_model.fit(X_scaled, y)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_scaled, y)

    models = {
        "CatBoost": {"model": cat_model, "scaler": scaler, "imputer": imputer},
        "XGBoost": {"model": xgb_model, "scaler": scaler, "imputer": imputer},
        "RandomForest": {"model": rf_model, "scaler": scaler, "imputer": imputer}
    }
    return models, features

models, features = train_models(merged_df)

# -----------------------------
# 3️⃣ Prediction function for DataFrame
# -----------------------------
def predict_df(df, model_name):
    model_info = models[model_name]
    model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    X = df[features].values
    X_scaled = scaler.transform(imputer.transform(X))

    df = df.copy()
    df['default_prob'] = model.predict(X_scaled)

    # ECL calculation
    if 'original_principal_amount_ususd' in df.columns:
        df['ECL'] = df['default_prob'] * df['original_principal_amount_ususd']
    else:
        df['ECL'] = df['default_prob']

    sectoral_pd = df.groupby('Sector')['default_prob'].mean().reset_index()
    sectoral_ecl = df.groupby('Sector')['ECL'].sum().reset_index()

    return df, sectoral_pd, sectoral_ecl

# -----------------------------
# 4️⃣ Streamlit Layout
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Model Selection", "Feature Input & Prediction", "Sectoral Analysis", "Stress Testing"])

# -----------------------------
# Homepage
# -----------------------------
if page == "Homepage":
    st.title("Credit Risk Analysis & ECL Modelling")
    st.markdown("""
    This app predicts:
    - Project-Level & Sectoral PD
    - Expected Credit Loss (ECL)
    - Stress testing scenarios
    """)
    st.write("Dataset preview:")
    st.dataframe(merged_df.head())

# -----------------------------
# Model Selection
# -----------------------------
elif page == "Model Selection":
    st.title("Select Regression Model")
    model_name = st.selectbox("Choose model", ["CatBoost", "XGBoost", "RandomForest"])
    st.write(f"Using model: **{model_name}**")

# -----------------------------
# Feature Input & Prediction
# -----------------------------
elif page == "Feature Input & Prediction":
    st.title("Predict using existing dataset")
    model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
    preds_df, _, _ = predict_df(merged_df, model_name)
    st.subheader("Project-level PD & ECL")
    st.dataframe(preds_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].head(10))

# -----------------------------
# Sectoral Analysis
# -----------------------------
elif page == "Sectoral Analysis":
    st.title("Sectoral PD & ECL")
    model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
    _, sectoral_pd, sectoral_ecl = predict_df(merged_df, model_name)
    st.subheader("Sectoral PD")
    st.dataframe(sectoral_pd)
    st.subheader("Sectoral ECL")
    st.dataframe(sectoral_ecl)

# -----------------------------
# Stress Testing Simulation
# -----------------------------
elif page == "Stress Testing":
    st.title("Stress Testing Scenario Simulation")
    model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])

    st.write("Define stress scenario (% change on features):")
    scenario_changes = {}
    for feature in features:
        scenario_changes[feature] = st.slider(f"{feature} change (%)", -50, 100, 0)

    if st.button("Apply Stress Scenario"):
        stressed_df = merged_df.copy()
        for feature, change in scenario_changes.items():
            stressed_df[feature] = stressed_df[feature] * (1 + change / 100)

        stressed_df, sectoral_pd, sectoral_ecl = predict_df(stressed_df, model_name)
        st.subheader("Sectoral PD under stress scenario")
        st.dataframe(sectoral_pd)
        st.subheader("Sectoral ECL under stress scenario")
        st.dataframe(sectoral_ecl)
        st.write("Project-level PD & ECL (first 10 rows)")
        st.dataframe(stressed_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].head(10))
