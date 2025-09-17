# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="India Credit Risk Dashboard", layout="wide")

# --------------------------
# Sidebar: File upload
# --------------------------
st.sidebar.title("Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# Load default data if no upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.success("File uploaded successfully!")
else:
    st.warning("Using default 'india_credit_risk.csv'")
    df = pd.read_csv("india_credit_risk.csv", parse_dates=True)

# --------------------------
# Preprocessing
# --------------------------
st.header("Data Preprocessing")

# Clean column names
df.columns = [
    col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("$", "usd").replace("'", "").replace(".", "")
    for col in df.columns
]

# Filter India
india_df = df[df['country___economy'].str.strip() == 'India'].copy()

# Drop useless columns
drop_cols = ['currency_of_commitment']
for col in drop_cols:
    if col in india_df.columns:
        india_df.drop(columns=[col], inplace=True)

# Convert dates
date_cols = ['end_of_period', 'first_repayment_date', 'last_repayment_date', 
             'agreement_signing_date', 'board_approval_date', 'effective_date_most_recent', 
             'closed_date_most_recent', 'last_disbursement_date']

for col in date_cols:
    if col in india_df.columns:
        india_df[col] = pd.to_datetime(india_df[col], errors='coerce')

india_df["origination_year"] = india_df["agreement_signing_date"].dt.year.astype("Int64")

# Numeric features
numeric_cols = [
    'interest_rate', 'original_principal_amount_ususd', 'cancelled_amount_ususd',
    'undisbursed_amount_ususd', 'disbursed_amount_ususd', 'repaid_to_ibrd_ususd',
    'due_to_ibrd_ususd','exchange_adjustment_ususd', 'borrowers_obligation_ususd', 'loans_held_ususd'
]
india_df[numeric_cols] = india_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
for col in numeric_cols:
    india_df[col] = india_df[col].apply(lambda x: np.nan if x < 0 else x)

# Loan status & default
active_statuses = ['REPAYING','DISBURSED','DISBURSING','DISBURSING&REPAYING',
                   'FULLY DISBURSED','FULLY TRANSFERRED','APPROVED','SIGNED','EFFECTIVE']

india_df['loan_status'] = india_df['loan_status'].astype('string').str.strip().str.upper()
india_df["is_active"] = india_df["loan_status"].isin(active_statuses).astype(int)

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

india_df["default_flag"] = india_df.apply(lambda row: encode_default_balanced(row["loan_status"], row["disbursed_amount_ususd"]), axis=1)

st.subheader("Dataset Overview")
st.write(india_df.head())

# --------------------------
# EAD, LGD, PD, ECL Calculations
# --------------------------
st.header("ECL Calculation")

usd_to_inr = 83

# Filter and sort
merged_df = india_df.copy()
merged_df = merged_df.sort_values(by=['project_name', 'end_of_period'])
merged_df = merged_df.dropna(subset=['project_name','borrowers_obligation_ususd'])

# Latest loans per project
latest_dates = merged_df.groupby('project_name')['end_of_period'].max().reset_index()
latest_loans = pd.merge(merged_df, latest_dates, on=['project_name','end_of_period'], how='inner')

# Weighted PD
pd_weighted = latest_loans.groupby('project_name').apply(
    lambda x: pd.Series({'PD': (x['borrowers_obligation_ususd'] * usd_to_inr * x['default_flag']).sum() /
                                  (x['borrowers_obligation_ususd'] * usd_to_inr).sum()})
).reset_index()

# EAD summary
ead_summary = latest_loans.groupby('project_name', as_index=False).agg(
    total_active_loans=('loan_number', 'count'),
    EAD_INR=('borrowers_obligation_ususd', lambda x: (x.sum() * usd_to_inr))
)

# LGD
merged_df['borrowers_obligation_inr'] = merged_df['borrowers_obligation_ususd'] * usd_to_inr
merged_df['repaid_to_ibrd_inr'] = merged_df['repaid_to_ibrd_ususd'] * usd_to_inr
merged_df['LGD'] = ((merged_df['borrowers_obligation_inr'] - merged_df['repaid_to_ibrd_inr']) / merged_df['borrowers_obligation_inr'])
merged_df.loc[merged_df['default_flag']==0,'LGD'] = 0

lgd_all_projects = merged_df.groupby('project_name', as_index=False).apply(
    lambda x: pd.Series({
        'EAD_INR': x['borrowers_obligation_inr'].sum(),
        'LGD': ((x['borrowers_obligation_inr'] * x['LGD']).sum()) / x['borrowers_obligation_inr'].sum()
    })
).reset_index(drop=True)

# Merge all
ecl_df = (ead_summary[['project_name','EAD_INR']]
          .merge(lgd_all_projects[['project_name','LGD']], on='project_name', how='left')
          .merge(pd_weighted, on='project_name', how='left'))

ecl_df['ECL_INR'] = ecl_df['EAD_INR'] * ecl_df['LGD'] * ecl_df['PD']
ecl_df = ecl_df.sort_values('ECL_INR', ascending=False)

st.subheader("Top 20 Projects by ECL")
st.write(ecl_df[['project_name','EAD_INR','LGD','PD','ECL_INR']].head(20))

# --------------------------
# Visualization
# --------------------------
st.header("Visualizations")

# Top ECL bar chart
top_plot = ecl_df.head(15)
fig, ax = plt.subplots(figsize=(12,6))
ax.barh(top_plot["project_name"], top_plot["ECL_INR"]/1e9, color="tomato")
ax.invert_yaxis()
ax.set_xlabel("ECL (INR Billion)")
ax.set_ylabel("Project")
ax.set_title("Top 15 Projects by Expected Credit Loss")
st.pyplot(fig)

# Sector analysis
def sector(name: str) -> str:
    n = str(name).upper()
    if any(w in n for w in ["ROAD","HIGHWAY","RAIL","TRANSPORT"]): return "Transport & Infrastructure"
    if any(w in n for w in ["POWER","ENERGY","SOLAR"]): return "Energy & Power"
    if any(w in n for w in ["WATER","IRRIGATION","DAM"]): return "Water & Irrigation"
    if any(w in n for w in ["URBAN","CITY","HOUSING"]): return "Urban Development & Housing"
    if any(w in n for w in ["AGRI","FARM","RURAL"]): return "Agriculture & Rural Development"
    if any(w in n for w in ["HEALTH","COVID","NUTRITION"]): return "Health & Social Protection"
    if any(w in n for w in ["EDUCATION","SCHOOL","TRAINING"]): return "Education & Skills"
    if any(w in n for w in ["FINANCE","MSME","BANK"]): return "Finance & Industry"
    if any(w in n for w in ["GOVERNANCE","SERVICE DELIVERY"]): return "Governance & Policy Reform"
    if any(w in n for w in ["CLIMATE","RESILIENT"]): return "Environment & Climate"
    if any(w in n for w in ["INNOVATE","TECH","ICT","DIGITAL"]): return "Technology & Innovation"
    if any(w in n for w in ["DISASTER","RECOVERY","RELIEF"]): return "Disaster Recovery & Emergency"
    return "Others"

merged_df["sector"] = merged_df["project_name"].apply(sector)
sector_ecl = merged_df.groupby("sector").apply(lambda x: (x['borrowers_obligation_ususd']*usd_to_inr*x['default_flag']).sum()).reset_index()
sector_ecl.columns = ["Sector","Total_ECL_INR"]

fig2, ax2 = plt.subplots(figsize=(12,6))
sns.barplot(data=sector_ecl, x="Total_ECL_INR", y="Sector", palette="viridis", ax=ax2)
ax2.set_xlabel("Total ECL (INR)")
ax2.set_ylabel("Sector")
st.pyplot(fig2)

# --------------------------
# Stress Testing
# --------------------------
st.header("Stress Testing Scenario")
stress_factor = st.slider("PD Stress Factor", 1.0, 3.0, 1.2, 0.05)

stress_ecl_df = ecl_df.copy()
stress_ecl_df['PD_stress'] = stress_ecl_df['PD'] * stress_factor
stress_ecl_df['ECL_stress_INR'] = stress_ecl_df['EAD_INR'] * stress_ecl_df['LGD'] * stress_ecl_df['PD_stress']

ecl_compare_df = ecl_df[['project_name','ECL_INR']].merge(
    stress_ecl_df[['project_name','ECL_stress_INR']],
    on='project_name'
).sort_values('ECL_stress_INR', ascending=False)

st.subheader("Top 20 Projects by Stressed ECL")
st.write(ecl_compare_df.head(20))

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.barh(ecl_compare_df.head(15)["project_name"], ecl_compare_df.head(15)["ECL_stress_INR"]/1e9, color="darkorange")
ax3.invert_yaxis()
ax3.set_xlabel("Stressed ECL (INR Billion)")
ax3.set_ylabel("Project")
ax3.set_title("Top 15 Projects by Stressed ECL")
st.pyplot(fig3)

st.success("ECL and stress testing calculations completed!")
