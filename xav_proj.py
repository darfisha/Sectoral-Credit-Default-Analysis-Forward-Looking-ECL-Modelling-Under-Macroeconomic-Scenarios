import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import statsmodels.api as sm
import shap

# Set page configuration
st.set_page_config(layout="wide", page_title="Credit Risk Analysis Dashboard")

# --- DATA LOADING AND PREPROCESSING ---
st.title("Credit Risk Analysis Dashboard")

@st.cache_data
def load_data():
    """
    Loads and preprocesses the credit risk and stress test data.
    This function is cached to prevent re-running on every interaction.
    """
    # Download data from Google Drive
    file_id = "1MVW1amhh9k3ksDsJkRo9ELvEwRplG0r2"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "credit_risk.csv"

    if not os.path.exists(output):
        st.info("Downloading credit risk data...")
        gdown.download(url, output, quiet=True, fuzzy=True)

    # Load and clean main dataframe
    date_cols = [
        "Agreement Signing Date", "Board Approval Date", "Closed Date (Most Recent)",
        "Effective Date (Most Recent)", "First Repayment Date",
        "Last Disbursement Date", "Last Repayment Date",
    ]
    df = pd.read_csv(output, parse_dates=date_cols, low_memory=False)

    df.columns = [col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("$", "usd").replace("'", "").replace(".", "") for col in df.columns]

    india_df = df[df['country___economy'].str.strip() == 'India'].copy()
    if 'currency_of_commitment' in india_df.columns:
        india_df.drop(columns=['currency_of_commitment'], inplace=True)

    date_cols = ['end_of_period', 'first_repayment_date', 'last_repayment_date',
                 'agreement_signing_date', 'board_approval_date',
                 'effective_date_most_recent', 'closed_date_most_recent',
                 'last_disbursement_date']
    for col in date_cols:
        if col in india_df.columns:
            india_df[col] = pd.to_datetime(india_df[col], errors='coerce')

    india_df["origination_year"] = india_df["agreement_signing_date"].dt.year.astype("Int64")
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd'
    ]
    india_df[numeric_cols] = india_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols:
        india_df[col] = india_df[col].apply(lambda x: np.nan if x < 0 else x)

    for col in india_df.select_dtypes(include="object").columns:
        india_df[col] = india_df[col].astype(str).str.strip()

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

    india_df["default_flag"] = india_df.apply(lambda row: encode_default_balanced(row["loan_status"], row["disbursed_amount_ususd"]), axis=1)
    
    # Load and clean stress data
    stress_df = pd.read_excel('GDP+CPI DATA.xls')
    stress_long = stress_df.set_index("Indicator Name").T.reset_index()
    stress_long.columns = ["year", "gdp_growth", "cpi_inflation"]
    stress_long["year"] = stress_long["year"].astype(int)
    
    # Merge datasets
    india_df["year"] = india_df["agreement_signing_date"].dt.year
    merged_df = pd.merge(india_df, stress_long, on="year", how="inner")
    merged_df = merged_df[(merged_df["year"] >= 2014) & (merged_df["year"] <= 2024)].copy()

    # Feature Engineering
    merged_df["loan_to_gdp_growth_ratio"] = (
        merged_df["original_principal_amount_ususd"] / (merged_df["gdp_growth"] * 1e9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    merged_df["repayment_ratio"] = (
        merged_df["repaid_to_ibrd_ususd"] / merged_df["disbursed_amount_ususd"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Define numeric columns for modeling
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd',
        "repayment_ratio", "loan_to_gdp_growth_ratio",
    ]

    # Preprocessing pipeline
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    # Chronological split
    train_df = merged_df[(merged_df["year"] >= 2014) & (merged_df["year"] <= 2020)].copy()
    val_df   = merged_df[(merged_df["year"] >= 2021) & (merged_df["year"] <= 2022)].copy()
    test_df  = merged_df[(merged_df["year"] >= 2023) & (merged_df["year"] <= 2024)].copy()

    X_train = train_df[numeric_cols].values
    y_train = train_df["default_flag"].values
    X_val   = val_df[numeric_cols].values if not val_df.empty else np.empty((0, len(numeric_cols)))
    y_val   = val_df["default_flag"].values if not val_df.empty else np.array([])
    X_test  = test_df[numeric_cols].values if not test_df.empty else np.empty((0, len(numeric_cols)))
    y_test  = test_df["default_flag"].values if not test_df.empty else np.array([])
    
    # Impute and scale
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed  = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled  = scaler.transform(X_test_imputed)
    
    return merged_df, X_train_scaled, y_train, X_test_scaled, y_test, numeric_cols, scaler, imputer

@st.cache_data
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trains and evaluates multiple machine learning models.
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "CatBoost": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        st.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append([name, acc, prec, rec, f1, auc])
        trained_models[name] = model

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    results_df = results_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    
    return results_df, trained_models

# --- UI LOGIC ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis", "Model Performance", "Risk & Stress Analysis"])

# Load data and models once
merged_df, X_train_scaled, y_train, X_test_scaled, y_test, numeric_cols, scaler, imputer = load_data()
results_df, trained_models = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

if page == "Data Overview":
    st.header("Data Specifications")
    st.write("This dashboard analyzes a credit risk dataset for India from 2014-2024. The dataset was preprocessed to clean and engineer features for a machine learning model.")
    
    st.subheader("Raw Data Sample")
    st.dataframe(merged_df.head())
    
    st.subheader("Dataset Shape and Columns")
    st.write(f"Dataset shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    st.text("Column Information:")
    buffer = st.text_area("DataFrame Info", height=300)
    merged_df.info(buf=buffer)
    st.session_state.df_info = buffer
    

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")

    # Missing values
    st.subheader("Missing Data Percentage")
    missing_percent = merged_df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    if not missing_percent.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_percent.index, y=missing_percent.values, ax=ax, palette="viridis")
        ax.set_title("Percentage of Missing Values")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)
    else:
        st.write("No missing values found in the dataset.")
        
    # Loan Status Distribution
    st.subheader("Loan Status Distribution")
    status_counts = merged_df["loan_status"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=status_counts.values, y=status_counts.index, ax=ax, palette="plasma")
    ax.set_title("Distribution of Loan Status")
    ax.set_xlabel("Count")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("Correlation matrix of key numeric features and the default flag.")
    numeric_cols = [
        'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
        'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
        'due_to_ibrd_ususd','exchange_adjustment_ususd',
        'borrowers_obligation_ususd', 'loans_held_ususd',
        "repayment_ratio", "loan_to_gdp_growth_ratio",
    ]
    corr_df = merged_df[numeric_cols + ['default_flag']]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Default Rate Over Years
    st.subheader("Default Rate Over Time")
    default_trend = merged_df.groupby('year')['default_flag'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='year', y='default_flag', data=default_trend, marker='o', ax=ax)
    ax.set_title('Default Rate Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Default Rate')
    st.pyplot(fig)
    
    # Scatter plots
    st.subheader("Feature Relationships")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(x='gdp_growth', y='loan_to_gdp_growth_ratio', hue='default_flag', data=merged_df, ax=ax1)
    ax1.set_title("Loan-to-GDP Ratio vs. GDP Growth")
    
    sns.histplot(data=merged_df, x='repayment_ratio', hue='default_flag', kde=True, ax=ax2)
    ax2.set_title("Distribution of Repayment Ratio")
    st.pyplot(fig)

elif page == "Model Performance":
    st.header("Model Performance & Metrics")
    
    st.subheader("Model Comparison Table")
    st.write("ROC-AUC is the primary metric for model selection.")
    st.dataframe(results_df)
    
    st.subheader("Best 2 Models")
    top_2_models = results_df.iloc[:2]
    st.write("The two best-performing models based on ROC-AUC are:")
    st.dataframe(top_2_models)
    
    # Confusion Matrix for top 2 models
    st.subheader("Confusion Matrix")
    top_2_names = top_2_models['Model'].tolist()
    
    for model_name in top_2_names:
        model = trained_models[model_name]
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

elif page == "Risk & Stress Analysis":
    st.header("Risk & Stress Analysis")

    # Re-run ECL and stress test parts
    
    def run_risk_analysis(df, trained_models, numeric_cols, scaler, imputer):
        # Predict probability of default
        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        X_all_scaled = scaler.transform(imputer.transform(df[numeric_cols].values))
        df['default_prob'] = best_model.predict_proba(X_all_scaled)[:, 1]

        # Sectoral analysis
        def sector_mapping(name: str) -> str:
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
        df["sector"] = df["project_name"].apply(sector_mapping)
        sector_analysis = df.groupby('sector').agg(avg_default_prob=('default_prob', 'mean')).reset_index()

        # ECL Calculation (before stress)
        usd_to_inr = 83
        df['borrowers_obligation_inr'] = df['borrowers_obligation_ususd'] * usd_to_inr
        df['repaid_to_ibrd_inr'] = df['repaid_to_ibrd_ususd'] * usd_to_inr
        df['LGD'] = ((df['borrowers_obligation_inr'] - df['repaid_to_ibrd_inr']) / df['borrowers_obligation_inr'])
        df.loc[df['default_flag'] == 0, 'LGD'] = 0
        df['ECL_baseline_calc'] = df['borrowers_obligation_inr'] * df['LGD'] * df['default_prob']
        
        project_ecl_baseline = df.groupby('project_name')['ECL_baseline_calc'].sum().reset_index()
        project_ecl_baseline.rename(columns={'ECL_baseline_calc': 'ECL_baseline'}, inplace=True)
        
        # Stress Test
        yearly = df.groupby('year').agg(
            pd_pred=('default_prob', 'mean'),
            gdp_growth=('gdp_growth', 'mean'),
            cpi_inflation=('cpi_inflation', 'mean')
        ).reset_index()
        yearly = yearly.dropna(subset=['gdp_growth', 'cpi_inflation']).sort_values('year')
        
        X = yearly[['gdp_growth', 'cpi_inflation']]
        X = sm.add_constant(X)
        y = yearly['pd_pred']
        model = sm.OLS(y, X).fit(cov_type='HC3')
        beta_gdp = model.params.get('gdp_growth', 0)
        beta_cpi = model.params.get('cpi_inflation', 0)
        
        delta_gdp = -3.0
        delta_cpi = 2.0
        
        df["pd_stressed"] = df["default_prob"] + abs(beta_gdp*delta_gdp) + abs(beta_cpi*delta_cpi)
        df["pd_stressed"] = df["pd_stressed"].clip(0,1)

        # Stressed ECL
        df['ECL_stressed_calc'] = df['borrowers_obligation_inr'] * df['LGD'] * df['pd_stressed']
        project_ecl_stressed = df.groupby('project_name')['ECL_stressed_calc'].sum().reset_index()
        project_ecl_stressed.rename(columns={'ECL_stressed_calc': 'ECL_stressed'}, inplace=True)

        # Merge for final analysis
        project_ecl = project_ecl_baseline.merge(project_ecl_stressed, on='project_name')
        project_ecl['Change_ECL'] = project_ecl['ECL_stressed'] - project_ecl['ECL_baseline']
        project_ecl['Change_ECL_%'] = 100 * project_ecl['Change_ECL'] / project_ecl['ECL_baseline']
        project_ecl = project_ecl.sort_values('ECL_stressed', ascending=False)
        
        return sector_analysis, project_ecl, df

    sector_analysis, project_ecl, df_with_risk_metrics = run_risk_analysis(merged_df.copy(), trained_models, numeric_cols, scaler, imputer)

    # --- Section: Project and Sectoral Analysis ---
    st.subheader("Project-level and Sectoral Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 15 Projects by Baseline ECL**")
        plot_df = project_ecl.sort_values("ECL_baseline", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="ECL_baseline", y="project_name", data=plot_df, palette="viridis", ax=ax)
        ax.set_title("Top 15 Projects by Expected Credit Loss (Baseline)")
        ax.set_xlabel("ECL (INR)")
        ax.set_ylabel("Project")
        st.pyplot(fig)
        
    with col2:
        st.write("**Average Default Probability by Sector**")
        sector_analysis_sorted = sector_analysis.sort_values('avg_default_prob', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='avg_default_prob', y='sector', data=sector_analysis_sorted, palette='Reds_r', ax=ax)
        ax.set_title("Average Default Probability by Sector")
        ax.set_xlabel("Average Default Probability")
        ax.set_ylabel("Sector")
        st.pyplot(fig)

    # --- Section: Stress Test Results ---
    st.subheader("Stress Test Results: Baseline vs. Stressed")
    st.write("The models were subjected to a stress scenario to see the impact on default probabilities and ECL.")
    
    st.markdown("##### Project-Level ECL: Baseline vs. Stressed")
    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(project_ecl[['project_name', 'ECL_baseline', 'ECL_stressed', 'Change_ECL_%']].head(10))
    with col4:
        st.write("A visual comparison of baseline vs. stressed ECL for the top 15 projects.")
        top_projects = project_ecl.head(15)
        fig, ax = plt.subplots(figsize=(12, 6))
        indices = np.arange(len(top_projects))
        bar_width = 0.4
        ax.barh(indices - bar_width/2, top_projects['ECL_baseline']/1e9, bar_width, label='Baseline ECL', color='skyblue')
        ax.barh(indices + bar_width/2, top_projects['ECL_stressed']/1e9, bar_width, label='Stressed ECL', color='salmon')
        ax.set_yticks(indices)
        ax.set_yticklabels(top_projects['project_name'])
        ax.set_xlabel('ECL (INR Billion)')
        ax.set_ylabel('Project')
        ax.set_title('Top 15 Projects: Baseline vs Stressed ECL')
        ax.legend()
        ax.invert_yaxis()
        st.pyplot(fig)

    st.markdown("##### Sector-Level PD: Baseline vs. Stressed")
    st.write("How the average default probability for each sector changes under the stress scenario.")
    
    sector_plot = df_with_risk_metrics.groupby('sector').agg(
        avg_pd_baseline=('default_prob', 'mean'),
        avg_pd_stressed=('pd_stressed', 'mean')
    ).reset_index().sort_values('avg_pd_stressed', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    indices = np.arange(len(sector_plot))
    bar_width = 0.35
    ax.bar(indices - bar_width/2, sector_plot['avg_pd_baseline'], bar_width, label='Baseline PD', color='#4C72B0')
    ax.bar(indices + bar_width/2, sector_plot['avg_pd_stressed'], bar_width, label='Stressed PD', color='#DD8452')
    ax.set_xticks(indices)
    ax.set_xticklabels(sector_plot['sector'], rotation=45, ha='right')
    ax.set_ylabel('Average Probability of Default')
    ax.set_title('Sectoral Average PD: Baseline vs Stressed')
    ax.legend()
    st.pyplot(fig)
