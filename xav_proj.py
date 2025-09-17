import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Sectoral ECL Analysis - Baseline vs Stressed")

# -------------------------------
# Step 0: Upload or load DataFrames
# -------------------------------
st.sidebar.header("Upload CSV files")
merged_file = st.sidebar.file_uploader("Upload merged_df CSV", type="csv")
ecl_file = st.sidebar.file_uploader("Upload ecl_compare_df CSV", type="csv")

if merged_file is not None:
    merged_df = pd.read_csv(merged_file)
    st.success("✅ merged_df loaded successfully!")
else:
    st.warning("⚠️ Please upload merged_df CSV to continue.")
    st.stop()

if ecl_file is not None:
    ecl_compare_df = pd.read_csv(ecl_file)
    st.success("✅ ecl_compare_df loaded successfully!")
else:
    st.warning("⚠️ Please upload ecl_compare_df CSV to continue.")
    st.stop()

# -------------------------------
# Step 1: Ensure sector info exists
# -------------------------------
if "sector" in merged_df.columns:
    project_sector_map = merged_df[['project_name','sector']].drop_duplicates()
elif "sector_mapping" in globals() and isinstance(sector_mapping, pd.DataFrame):
    project_sector_map = sector_mapping[['project_name','sector']].drop_duplicates()
else:
    st.error("❌ Could not find project → sector mapping. Provide a DataFrame with columns ['project_name','sector'].")
    st.stop()

# -------------------------------
# Step 2: Merge sector info into ECL data
# -------------------------------
if 'project_name' not in ecl_compare_df.columns:
    st.error("❌ ecl_compare_df must contain the column 'project_name'.")
    st.stop()

ecl_sector_df = ecl_compare_df.merge(project_sector_map, on='project_name', how='left')

# Automatically fill missing sectors with 'Unknown'
missing_count = ecl_sector_df['sector'].isna().sum()
if missing_count > 0:
    st.warning(f"⚠️ {missing_count} projects had no sector mapping. Assigning 'Unknown'.")
    ecl_sector_df['sector'] = ecl_sector_df['sector'].fillna('Unknown')

# -------------------------------
# Step 3: Aggregate at sector level
# -------------------------------
for col in ['EAD_INR','ECL_Base_INR','ECL_Stressed_INR']:
    if col not in ecl_sector_df.columns:
        st.error(f"❌ Column '{col}' not found in ecl_compare_df.")
        st.stop()

sector_summary = (
    ecl_sector_df
    .groupby('sector')
    .agg({
        'EAD_INR':'sum',
        'ECL_Base_INR':'sum',
        'ECL_Stressed_INR':'sum'
    })
    .reset_index()
)

# -------------------------------
# Step 4: Calculate changes
# -------------------------------
sector_summary['Change_INR'] = sector_summary['ECL_Stressed_INR'] - sector_summary['ECL_Base_INR']
sector_summary['Change_%'] = (
    (sector_summary['Change_INR'] / sector_summary['ECL_Base_INR'].replace(0, np.nan)) * 100
).round(2)

# -------------------------------
# Step 5: Human-readable formatting
# -------------------------------
def human_readable(num):
    if num >= 1e9:
        return f"{num/1e9:.0f} Billion"
    elif num >= 1e6:
        return f"{num/1e6:.0f} Million"
    elif num >= 1e3:
        return f"{num/1e3:.0f} Thousand"
    else:
        return f"{num:.0f}"

sector_display = sector_summary.copy()
for col in ['EAD_INR','ECL_Base_INR','ECL_Stressed_INR','Change_INR']:
    sector_display[col] = sector_display[col].apply(human_readable)

st.subheader("✅ Sectoral ECL - Before vs After Stress Testing")
st.dataframe(sector_display[['sector','EAD_INR','ECL_Base_INR','ECL_Stressed_INR','Change_INR','Change_%']])

# -------------------------------
# Step 6: Visualization
# -------------------------------
plot_df = sector_summary.sort_values('ECL_Stressed_INR', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(14,7))
bar_width = 0.35
index = np.arange(len(plot_df))

# Baseline bars
ax.barh(index - bar_width/2,
        plot_df['ECL_Base_INR']/1e9,
        bar_width, label="Baseline ECL", color='steelblue')

# Stressed bars
ax.barh(index + bar_width/2,
        plot_df['ECL_Stressed_INR']/1e9,
        bar_width, label="Stressed ECL", color='crimson')

ax.set_yticks(index)
ax.set_yticklabels(plot_df['sector'])
ax.invert_yaxis()
ax.set_title("Top 10 Sectors: Baseline vs Stressed ECL", fontsize=14)
ax.set_xlabel("ECL (INR Billion)")
ax.set_ylabel("Sector")
ax.legend()
plt.tight_layout()

# Add value labels
for i, (base, stress) in enumerate(zip(plot_df['ECL_Base_INR'], plot_df['ECL_Stressed_INR'])):
    ax.text(base/1e9, i - bar_width/2, human_readable(base), va='center', ha='left', fontsize=8)
    ax.text(stress/1e9, i + bar_width/2, human_readable(stress), va='center', ha='left', fontsize=8)

st.pyplot(fig)
