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

# Ensure Project and Sector columns exist
if 'ProjectName' not in merged_df.columns:
    merged_df['ProjectName'] = ["Project_" + str(i) for i in range(len(merged_df))]
if 'Sector' not in merged_df.columns:
    merged_df['Sector'] = ["Sector_" + str(i % 3 + 1) for i in range(len(merged_df))]

# -----------------------------
# 2️⃣ Train models once (cached)
# -----------------------------
@st.cache_resource
def train_models(df):
    # numeric features only, exclude target
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

# Train once
models, features = train_models(merged_df)

# -----------------------------
# 3️⃣ Prediction function for new input
# -----------------------------
def predict_input(input_df, model_name):
    model_info = models[model_name]
    model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    X_input = input_df[features].values
    X_scaled = scaler.transform(imputer.transform(X_input))

    input_df = input_df.copy()
    input_df['default_prob'] = model.predict(X_scaled)

    # ECL calculation
    if 'original_principal_amount_ususd' in input_df.columns:
        input_df['ECL'] = input_df['default_prob'] * input_df['original_principal_amount_ususd']
    else:
        input_df['ECL'] = input_df['default_prob']

    # Use actual ProjectName and Sector from the input_df if present
    if 'ProjectName' not in input_df.columns:
        input_df['ProjectName'] = ["Project_" + str(i) for i in range(len(input_df))]
    if 'Sector' not in input_df.columns:
        input_df['Sector'] = ["Sector_" + str(i % 3 + 1) for i in range(len(input_df))]

    sectoral_pd = input_df.groupby('Sector')['default_prob'].mean().reset_index()
    sectoral_ecl = input_df.groupby('Sector')['ECL'].sum().reset_index()

    return input_df, sectoral_pd, sectoral_ecl

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
    This web app allows you to:
    - Predict Project-Level and Sectoral Probability of Default (PD)
    - Compute Expected Credit Loss (ECL)
    - Perform sectoral stress testing
    - Compare CatBoost, XGBoost, and RandomForest regressors
    """)
    st.write("Dataset preview:")
    st.dataframe(merged_df.head())

# -----------------------------
# Model Selection
# -----------------------------
elif page == "Model Selection":
    st.title("Select Regression Model")
    model_name = st.selectbox("Choose model", ["CatBoost", "XGBoost", "RandomForest"])
    st.write(f"You selected **{model_name}**.")

# -----------------------------
# Feature Input & Prediction (Dynamic)
# -----------------------------
elif page == "Feature Input & Prediction":
    st.title("Dynamic Feature Input for Prediction")
    features_input = {}

    with st.form("feature_form"):
        st.write("Enter feature values:")
        for feature in features:
            features_input[feature] = st.number_input(feature, value=float(merged_df[feature].mean()))
        # Optionally input project name and sector
        project_name = st.text_input("Project Name", value="New Project")
        sector_name = st.text_input("Sector", value="New Sector")
        model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([features_input])
        input_df['ProjectName'] = project_name
        input_df['Sector'] = sector_name
        preds_df, _, _ = predict_input(input_df, model_name)
        st.write("Predicted default probability for input:")
        st.dataframe(preds_df[['ProjectName', 'Sector', 'default_prob', 'ECL']])

# -----------------------------
# Sectoral Analysis
# -----------------------------
elif page == "Sectoral Analysis":
    st.title("Sectoral PD & ECL")
    model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
    preds_df, sectoral_pd, sectoral_ecl = predict_input(merged_df, model_name)
    st.subheader("Sectoral PD")
    st.dataframe(sectoral_pd)
    st.subheader("Sectoral ECL")
    st.dataframe(sectoral_ecl)
    st.subheader("Project-level PD & ECL")
    st.dataframe(preds_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].head(10))

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

        stressed_df, sectoral_pd, sectoral_ecl = predict_input(stressed_df, model_name)
        st.subheader("Sectoral PD under stress scenario")
        st.dataframe(sectoral_pd)
        st.subheader("Sectoral ECL under stress scenario")
        st.dataframe(sectoral_ecl)
        st.write("Project-level PD (first 10 rows):")
        st.dataframe(stressed_df[['ProjectName', 'Sector', 'default_prob', 'ECL']].head(10))
