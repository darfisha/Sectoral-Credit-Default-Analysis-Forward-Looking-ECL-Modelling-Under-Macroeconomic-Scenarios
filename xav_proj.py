import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1️⃣ Load CSV from GitHub
# -----------------------------
GITHUB_CSV_URL = "https://raw.githubusercontent.com/darfisha/Sectoral-Credit-Default-Analysis-Forward-Looking-ECL-Modelling-Under-Macroeconomic-Scenarios/refs/heads/main/merged_df.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

merged_df = load_data(GITHUB_CSV_URL)

# -----------------------------
# 2️⃣ Train models (cached)
# -----------------------------
@st.cache_resource
def train_models(df):
    features = df.select_dtypes(include='number').columns.drop('default_flag').tolist()
    X = df[features].values
    y = df['default_flag'].values

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    catboost_model = CatBoostClassifier(iterations=100, verbose=0)
    catboost_model.fit(X_scaled, y)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_scaled, y)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)

    models = {
        "CatBoost": {"model": catboost_model, "scaler": scaler, "imputer": imputer},
        "XGBoost": {"model": xgb_model, "scaler": scaler, "imputer": imputer},
        "RandomForest": {"model": rf_model, "scaler": scaler, "imputer": imputer}
    }
    return models, features

# -----------------------------
# 3️⃣ Predict function (cached)
# -----------------------------
@st.cache_data
def predict(df, model_name):
    models, features = train_models(df)
    model_info = models[model_name]
    model = model_info["model"]
    scaler = model_info["scaler"]
    imputer = model_info["imputer"]

    X_all = df[features].values
    X_all_scaled = scaler.transform(imputer.transform(X_all))

    df = df.copy()
    df['default_prob'] = model.predict_proba(X_all_scaled)[:, 1]

    if 'original_principal_amount_ususd' in df.columns:
        df['ECL'] = df['default_prob'] * df['original_principal_amount_ususd']
    else:
        df['ECL'] = df['default_prob']

    if 'Sector' not in df.columns:
        df['Sector'] = "Sector_" + (df.index % 3 + 1).astype(str)

    sectoral_pd = df.groupby('Sector')['default_prob'].mean().reset_index()
    sectoral_ecl = df.groupby('Sector')['ECL'].sum().reset_index()

    return df, sectoral_pd, sectoral_ecl, features

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
        - Evaluate Project-Level and Sectoral PD
        - Compute Expected Credit Losses (ECL)
        - Perform sectoral stress testing
        - Compare CatBoost, XGBoost, and RandomForest predictions
    """)
    st.write("Dataset preview:")
    st.dataframe(merged_df.head())

# -----------------------------
# Model Selection
# -----------------------------
elif page == "Model Selection":
    st.title("Select ML Model")
    model_name = st.selectbox("Choose model", ["CatBoost", "XGBoost", "RandomForest"])
    st.write(f"You selected **{model_name}**.")

# -----------------------------
# Feature Input & Prediction (Dynamic)
# -----------------------------
elif page == "Feature Input & Prediction":
    st.title("Dynamic Feature Input for Prediction")
    _, _, _, features = predict(merged_df, "CatBoost")  # get features
    features_input = {}

    with st.form("feature_form"):
        st.write("Enter feature values:")
        for feature in features:
            features_input[feature] = st.number_input(feature, value=float(merged_df[feature].mean()))
        model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([features_input])
        full_df = pd.concat([merged_df, input_df], ignore_index=True)
        preds_df, _, _, _ = predict(full_df, model_name)
        st.write("Predicted default probability for input:")
        st.write(preds_df.tail(1)[['default_prob']])

# -----------------------------
# Sectoral Analysis
# -----------------------------
elif page == "Sectoral Analysis":
    st.title("Sectoral PD & ECL")
    model_name = st.selectbox("Select model", ["CatBoost", "XGBoost", "RandomForest"])
    preds_df, sectoral_pd, sectoral_ecl, _ = predict(merged_df, model_name)
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
    preds_df, _, _, features = predict(merged_df, model_name)

    st.write("Define stress scenario (% change on features):")
    scenario_changes = {}
    for feature in features:
        scenario_changes[feature] = st.slider(f"{feature} change (%)", -50, 100, 0)

    if st.button("Apply Stress Scenario"):
        stressed_df = preds_df.copy()
        for feature, change in scenario_changes.items():
            stressed_df[feature] = stressed_df[feature] * (1 + change / 100)

        stressed_df, sectoral_pd, sectoral_ecl, _ = predict(stressed_df, model_name)
        st.subheader("Sectoral PD under stress scenario")
        st.dataframe(sectoral_pd)
        st.subheader("Sectoral ECL under stress scenario")
        st.dataframe(sectoral_ecl)
        st.write("Project-level PD (first 10 rows):")
        st.dataframe(stressed_df[['default_prob']].head(10))
