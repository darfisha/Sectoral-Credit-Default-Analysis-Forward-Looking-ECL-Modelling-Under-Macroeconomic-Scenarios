import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Step 0: Load required DataFrames
# -------------------------------
data_files = {
    'merged_df': 'merged_df.csv',           # Path to your merged_df CSV
    'ecl_compare_df': 'ecl_compare_df.csv'  # Path to your ECL comparison CSV
}

for df_name, path in data_files.items():
    if df_name not in globals():
        if os.path.exists(path):
            globals()[df_name] = pd.read_csv(path)
            print(f"✅ Loaded {df_name} from {path}")
        else:
            raise FileNotFoundError(f"❌ {df_name} not found in globals and file '{path}' does not exist. Please provide the file.")

# -------------------------------
# Step 1: Ensure sector info exists
# -------------------------------
if "sector" in merged_df.columns:
    project_sector_map = merged_df[['project_name','sector']].drop_duplicates()
elif "sector_mapping" in globals() and isinstance(sector_mapping, pd.DataFrame):
    project_sector_map = sector_mapping[['project_name','sector']].drop_duplicates()
else:
    raise ValueError("❌ Could not find project → sector mapping. Provide a DataFrame with columns ['project_name','sector'].")

# -------------------------------
# Step 2: Merge sector info into ECL data
# -------------------------------
if 'project_name' not in ecl_compare_df.columns:
    raise ValueError("❌ ecl_compare_df must contain the column 'project_name'.")

ecl_sector_df = ecl_compare_df.merge(project_sector_map, on='project_name', how='left')

# Automatically fill missing sectors with 'Unknown'
missing_count = ecl_sector_df['sector'].isna().sum()
if missing_count > 0:
    print(f"⚠️ Warning: {missing_count} projects had no sector mapping. Assigning 'Unknown'.")
    ecl_sector_df['sector'] = ecl_sector_df['sector'].fillna('Unknown')

# -------------------------------
# Step 3: Aggregate at sector level
# -------------------------------
for col in ['EAD_INR','ECL_Base_INR','ECL_Stressed_INR']:
    if col not in ecl_sector_df.columns:
        raise ValueError(f"❌ Column '{col}' not found in ecl_compare_df.")

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

print("\n✅ Sectoral ECL - Before vs After Stress Testing\n")
print(sector_display[['sector','EAD_INR','ECL_Base_INR','ECL_Stressed_INR','Change_INR','Change_%']])

# -------------------------------
# Step 6: Visualization
# -------------------------------
plot_df = sector_summary.sort_values('ECL_Stressed_INR', ascending=False).head(10)

plt.figure(figsize=(14,7))
bar_width = 0.35
index = np.arange(len(plot_df))

plt.barh(index - bar_width/2,
         plot_df['ECL_Base_INR']/1e9,
         bar_width, label="Baseline ECL", color='steelblue')

plt.barh(index + bar_width/2,
         plot_df['ECL_Stressed_INR']/1e9,
         bar_width, label="Stressed ECL", color='crimson')

plt.yticks(index, plot_df['sector'])
plt.gca().invert_yaxis()
plt.title("Top 10 Sectors: Baseline vs Stressed ECL", fontsize=14)
plt.xlabel("ECL (INR Billion)")
plt.ylabel("Sector")
plt.legend()
plt.tight_layout()

for i, (base, stress) in enumerate(zip(plot_df['ECL_Base_INR'], plot_df['ECL_Stressed_INR'])):
    plt.text(base/1e9, i - bar_width/2, human_readable(base), va='center', ha='left', fontsize=8)
    plt.text(stress/1e9, i + bar_width/2, human_readable(stress), va='center', ha='left', fontsize=8)

plt.show()
