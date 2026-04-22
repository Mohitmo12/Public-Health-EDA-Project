# ============================================================
# PUBLIC HEALTH SURVEILLANCE DATASET — EDA PROJECT
# Dataset : Public Health Surveillance
# Goal    : Analyze disease spread, vaccination impact,
#           environmental factors & severity patterns
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency, f_oneway

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
def variance_inflation_factor(X, idx):
    """Manual VIF: regress feature idx on all others, compute 1/(1-R²)."""
    from sklearn.linear_model import LinearRegression as _LR
    y_vif = X[:, idx]
    X_vif = np.delete(X, idx, axis=1)
    r2 = _LR().fit(X_vif, y_vif).score(X_vif, y_vif)
    return 1.0 / (1 - r2) if r2 < 1 else float('inf')

import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────
# 1. DATA LOADING & FIRST LOOK
# ─────────────────────────────────────────────

df_original = pd.read_csv("public_health_surveillance_dataset.csv")
df = df_original.copy()

print("=" * 60)
print("  PUBLIC HEALTH SURVEILLANCE — EDA PROJECT")
print("=" * 60)

df.info()
print("\nShape:", df.shape)

print("\n── Descriptive Statistics ──")
print(df.describe().round(2))

print("\n── Missing Values ──")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# ─────────────────────────────────────────────
# 2. DATA CLEANING + FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Fill missing text columns
df['Medical_History']   = df['Medical_History'].fillna("Unknown")
df['Reported_Symptoms'] = df['Reported_Symptoms'].fillna("Not Reported")
df['Diagnosis']         = df['Diagnosis'].fillna("Not Diagnosed")

# Convert dates
df['Date_of_Onset']           = pd.to_datetime(df['Date_of_Onset'])
df['Date_of_Data_Collection'] = pd.to_datetime(df['Date_of_Data_Collection'])

# Feature Engineering
df['Month']    = df['Date_of_Onset'].dt.month
df['Month_Name'] = df['Date_of_Onset'].dt.strftime('%b')
df['Year']     = df['Date_of_Onset'].dt.year
df['Day_Name'] = df['Date_of_Onset'].dt.day_name()

# Composite environmental risk score
df['Env_Risk_Score'] = df['AQI'] * df['Transmission_Rate']

# Binary severity flag (Severe = 1)
df['Is_Severe'] = (df['Disease_Severity'] == 'Severe').astype(int)

# Binary: positive test
df['Is_Positive'] = (df['Testing_Results'] == 'Positive').astype(int)

# Age bins
df['Age_Group'] = pd.cut(df['Age'],
                         bins=[0,17,35,60,100],
                         labels=['Child (0-17)', 'Young Adult (18-35)',
                                 'Adult (36-60)', 'Senior (61+)'])

print("\nCleaning complete. New columns preview:")
print(df[['Date_of_Onset','Month','Age_Group','Env_Risk_Score','Is_Severe']].head(5))


# ─────────────────────────────────────────────
# 3. CORRELATION HEATMAP
# ─────────────────────────────────────────────

num_cols = ['Age','Temperature','AQI','Humidity',
            'Transmission_Rate','Mortality_Rate',
            'Case_Fatality_Ratio','Daily_New_Cases',
            'Resource_Utilization']

plt.figure(figsize=(11, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5, square=True)
plt.title("Correlation Heatmap — Numerical Features", fontsize=14)
plt.tight_layout()
plt.savefig("plots/plot_00_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# OBJECTIVE 1 — Vaccination Impact on Transmission
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  OBJECTIVE 1 — Vaccination Impact")
print("=" * 55)

vacc_map = {0: 'Non-Vaccinated', 1: 'Vaccinated'}
df['Vacc_Label'] = df['Vaccination_Status'].map(vacc_map)

vacc_effect = df.groupby('Vacc_Label')['Transmission_Rate'].mean().sort_values()
print(vacc_effect)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar plot
sns.barplot(x=vacc_effect.index, y=vacc_effect.values,
            hue=vacc_effect.index, palette='coolwarm', ax=axes[0])
axes[0].set_title("Avg Transmission Rate: Vaccinated vs Non-Vaccinated", fontsize=13)
axes[0].set_ylabel("Avg Transmission Rate")
axes[0].set_xlabel("Vaccination Status")

# Box plot
sns.boxplot(data=df, x='Vacc_Label', y='Transmission_Rate',
            hue='Vacc_Label', palette='Set2', ax=axes[1])
axes[1].set_title("Transmission Distribution by Vaccination Status", fontsize=13)
axes[1].set_ylabel("Transmission Rate")
axes[1].set_xlabel("Vaccination Status")

plt.tight_layout()
plt.savefig("plots/plot_01_vaccination.png", dpi=150, bbox_inches='tight')

plt.show()

# Also check hesitancy
hesitancy_tx = df.groupby('Vaccination_Hesitancy')['Transmission_Rate'].mean()
plt.figure(figsize=(6, 4))
sns.barplot(x=hesitancy_tx.index, y=hesitancy_tx.values,
            hue=hesitancy_tx.index, palette='Reds')
plt.title("Transmission Rate by Vaccination Hesitancy")
plt.ylabel("Avg Transmission Rate")
plt.savefig("plots/plot_01b_hesitancy.png", dpi=150, bbox_inches='tight')
plt.show()

print("""
Insight:
  Vaccinated individuals show a lower average transmission rate, 
  indicating measurable effectiveness of vaccination in reducing disease spread. 
  Those with vaccine hesitancy also show higher transmission, 
  reinforcing the value of vaccination campaigns.
""")


# ─────────────────────────────────────────────
# OBJECTIVE 2 — Environmental Factors (AQI & Humidity)
# ─────────────────────────────────────────────

print("=" * 55)
print("  OBJECTIVE 2 — Environmental Impact")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# AQI bins vs transmission
df['AQI_Bin'] = pd.cut(df['AQI'], bins=[0,50,100,150,200,300],
                        labels=['Good','Moderate','USG','Unhealthy','Hazardous'])
aqi_tx = df.groupby('AQI_Bin', observed=True)['Transmission_Rate'].mean()
sns.barplot(x=aqi_tx.index.astype(str), y=aqi_tx.values,
            hue=aqi_tx.index.astype(str), palette='YlOrRd', ax=axes[0])
axes[0].set_title("AQI Category vs Avg Transmission Rate", fontsize=13)
axes[0].set_xlabel("AQI Category")
axes[0].set_ylabel("Avg Transmission Rate")
axes[0].tick_params(axis='x', rotation=20)

# Humidity bins vs transmission
df['Hum_Bin'] = pd.cut(df['Humidity'], bins=8)
hum_tx = df.groupby('Hum_Bin', observed=True)['Transmission_Rate'].mean()
sns.lineplot(x=hum_tx.index.astype(str), y=hum_tx.values,
             marker='o', color='steelblue', linewidth=2, ax=axes[1])
axes[1].set_title("Humidity vs Transmission Rate Trend", fontsize=13)
axes[1].set_xlabel("Humidity Range (%)")
axes[1].set_ylabel("Avg Transmission Rate")
axes[1].tick_params(axis='x', rotation=35)

plt.tight_layout()
plt.savefig("plots/plot_02_environment.png", dpi=150, bbox_inches='tight')

plt.show()

print("""
Insight:
  Higher AQI (poorer air quality) is associated with increased transmission rates,
  likely due to airborne pathogen facilitation. Humidity also shows a trend —
  moderate-to-high humidity environments exhibit elevated transmission.
""")

# ==============================
# OBJECTIVE 3 — Age vs Severity
# ==============================

print("=" * 55)
print("  OBJECTIVE 3 — Age vs Disease Severity")
print("=" * 55)

sev_order = ['Mild', 'Moderate', 'Severe']


fig, ax = plt.subplots(figsize=(8,5))

# Data prep
age_sev = df.groupby(['Age_Group', 'Disease_Severity'],
                     observed=True).size().unstack(fill_value=0)

age_sev_pct = age_sev.div(age_sev.sum(axis=1), axis=0) * 100

# Plot
age_sev_pct[sev_order].plot(
    kind='bar',
    stacked=True,
    colormap='RdYlGn_r',
    ax=ax
)

ax.set_title("Disease Severity (%) by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("% of Patients")

plt.tight_layout()


plt.savefig("plots/plot_03_age_severity.png", dpi=150, bbox_inches='tight')

plt.show()

# ─────────────────────────────────────────────
# OBJECTIVE 4 — Social Behavior & Infection Risk
# ─────────────────────────────────────────────

print("=" * 55)
print("  OBJECTIVE 4 — Social Behavior & Infection Risk")
print("=" * 55)

activity_order = ['Low', 'Medium', 'High']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Social activity vs transmission
social_tx = df.groupby('Social_Activity')['Transmission_Rate'].mean().reindex(activity_order)
sns.barplot(x=social_tx.index, y=social_tx.values,
            hue=social_tx.index, palette='viridis', ax=axes[0])
axes[0].set_title("Social Activity vs Avg Transmission Rate", fontsize=13)
axes[0].set_xlabel("Social Activity Level")
axes[0].set_ylabel("Avg Transmission Rate")

# Compliance vs transmission
comp_tx = df.groupby('Compliance_with_Health_Guidelines')['Transmission_Rate'].mean()
sns.barplot(x=['Non-Compliant','Compliant'], y=comp_tx.values,
            hue=['Non-Compliant','Compliant'], palette='Set3', ax=axes[1])
axes[1].set_title("Health Guideline Compliance vs Transmission Rate", fontsize=13)
axes[1].set_xlabel("Compliance")
axes[1].set_ylabel("Avg Transmission Rate")

plt.tight_layout()
plt.savefig("plots/plot_04_social.png", dpi=150, bbox_inches='tight')
plt.show()

# Risk level distribution by location
risk_loc = df.groupby(['Location','Infection_Risk_Level']).size().unstack(fill_value=0)
risk_loc_pct = risk_loc.div(risk_loc.sum(axis=1), axis=0) * 100
risk_loc_pct[['Low Risk','Medium Risk','High Risk']].plot(kind='bar', colormap='RdYlGn_r', figsize=(8,5))
plt.title("Infection Risk Level by Location Type", fontsize=13)
plt.xlabel("Location")
plt.ylabel("% of Cases")
plt.xticks(rotation=0)
plt.legend(title='Risk Level')
plt.tight_layout()
plt.savefig("plots/plot_04b_risk_location.png", dpi=150, bbox_inches='tight')
plt.show()

print("""
Insight:
  High social activity strongly correlates with elevated transmission.
  Compliance with health guidelines is clearly linked to lower transmission,
  underscoring the effectiveness of public health interventions.
  Urban areas show a higher proportion of Medium/High Risk cases 
  due to greater population density.
""")


# ─────────────────────────────────────────────
# OBJECTIVE 5 — Time-Based Trends
# ─────────────────────────────────────────────

print("=" * 55)
print("  OBJECTIVE 5 — Time-Based Trends")
print("=" * 55)

df_indexed = df.set_index('Date_of_Onset').copy()

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

# Weekly new cases
df_indexed['Daily_New_Cases'].resample('W').mean().plot(
    ax=axes[0], color='steelblue', linewidth=1.8)
axes[0].set_title("Weekly Average — Daily New Cases", fontsize=13)
axes[0].set_ylabel("Avg Daily New Cases")
axes[0].set_xlabel("")

# Monthly transmission rate
monthly_tx = df_indexed['Transmission_Rate'].resample('ME').mean()
monthly_tx.plot(ax=axes[1], color='tomato', marker='o', linewidth=1.8)
axes[1].set_title("Monthly Average — Transmission Rate", fontsize=13)
axes[1].set_ylabel("Avg Transmission Rate")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig("plots/plot_05_timeseries.png", dpi=150, bbox_inches='tight')
plt.show()


# Month-wise seasonal pattern
month_order = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

month_tx = df.groupby('Month_Name')['Transmission_Rate'].mean().reindex(month_order)

plt.figure(figsize=(10,4))

sns.lineplot(
    x=month_tx.index,
    y=month_tx.values,
    marker='o',
    color='darkorange',
    linewidth=2
)

plt.title("Average Transmission Rate by Month (Seasonal Pattern)")
plt.xlabel("Month")
plt.ylabel("Avg Transmission Rate")

plt.tight_layout()


plt.savefig("plots/plot_05b_seasonal.png", dpi=150, bbox_inches='tight')

plt.show()


# ─────────────────────────────────────────────
# OBJECTIVE 6 — Descriptive Statistics
# ─────────────────────────────────────────────

print("=" * 55)
print("  OBJECTIVE 6 — Descriptive Statistics")
print("=" * 55)

stat_cols = ['Age','Temperature','AQI','Humidity','Transmission_Rate',
             'Mortality_Rate','Case_Fatality_Ratio',
             'Daily_New_Cases','Resource_Utilization']

print("\nDETAILED DESCRIPTIVE STATISTICS")
print(df[stat_cols].describe().round(3).to_string())


# ─────────────────────────────────────────────
# OBJECTIVE 7 — STATISTICAL TESTS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  OBJECTIVE 7 — STATISTICAL TESTS")
print("=" * 55)

# ── T-TEST 1: Vaccination vs Transmission ──────────────────
vacc     = df[df['Vaccination_Status'] == 1]['Transmission_Rate']
non_vacc = df[df['Vaccination_Status'] == 0]['Transmission_Rate']
t1, p1   = ttest_ind(vacc, non_vacc)

print("\n─" * 28)
print("  T-TEST 1: Vaccination Effect on Transmission Rate")
print("─" * 28)
print(f"  H0: Vaccination has no effect on Transmission Rate")
print(f"  H1: Vaccination significantly reduces Transmission Rate")
print(f"\n  Vaccinated Mean     : {vacc.mean():.4f}")
print(f"  Non-Vaccinated Mean : {non_vacc.mean():.4f}")
print(f"  Difference          : {vacc.mean() - non_vacc.mean():.4f}")
print(f"  T-statistic         : {t1:.4f}")
print(f"  p-value             : {p1:.4f}")
print()
if p1 < 0.05:
    print("  ✔ Reject H0: Vaccination significantly affects transmission rate")
else:
    print("  ✘ Cannot Reject H0: No significant difference")

# ── T-TEST 2: Chronic Conditions vs Mortality ──────────────
chronic     = df[df['Chronic_Conditions'] == 1]['Mortality_Rate']
no_chronic  = df[df['Chronic_Conditions'] == 0]['Mortality_Rate']
t2, p2      = ttest_ind(chronic, no_chronic)

print("\n─" * 28)
print("  T-TEST 2: Chronic Conditions Effect on Mortality Rate")
print("─" * 28)
print(f"  H0: Chronic conditions have no effect on mortality")
print(f"  H1: Chronic conditions significantly increase mortality")
print(f"\n  Chronic Conditions Mean    : {chronic.mean():.4f}")
print(f"  No Chronic Conditions Mean : {no_chronic.mean():.4f}")
print(f"  T-statistic                : {t2:.4f}")
print(f"  p-value                    : {p2:.4f}")
print()
if p2 < 0.05:
    print("  ✔ Reject H0: Chronic conditions significantly affect mortality")
else:
    print("  ✘ Cannot Reject H0: No significant difference")

# ── CHI-SQUARE: Gender vs Infection Risk ──────────────────
table     = pd.crosstab(df['Gender'], df['Infection_Risk_Level'])
chi2, p3, dof, expected = chi2_contingency(table)

print("\n─" * 28)
print("  CHI-SQUARE: Gender vs Infection Risk Level")
print("─" * 28)
print(f"  H0: Gender and Infection Risk Level are independent")
print(f"  H1: There is an association between Gender and Infection Risk Level")
print(f"\n  Chi2 Statistic : {chi2:.4f}")
print(f"  Degrees of Freedom : {dof}")
print(f"  p-value        : {p3:.4f}")
print()
if p3 < 0.05:
    print("  ✔ Reject H0: Significant association between Gender and Infection Risk")
else:
    print("  ✘ Cannot Reject H0: Gender and Risk Level are independent")


# ─────────────────────────────────────────────
# OBJECTIVE 8 — KEY RISK FACTOR ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  OBJECTIVE 8 — KEY RISK FACTOR ANALYSIS")
print("=" * 55)

# Compare High Risk vs Low/Medium Risk
risk_groups = df.groupby('Infection_Risk_Level')

# Key variables to analyze
factors = ['Age', 'AQI', 'Humidity', 'Transmission_Rate',
           'Resource_Utilization']

risk_summary = risk_groups[factors].mean().round(3)
print("\nAverage Values by Risk Level:\n")
print(risk_summary)


# ── Visualization ─────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

sns.barplot(x='Infection_Risk_Level', y='AQI', data=df,
            palette='Reds', ax=axes[0,0])
axes[0,0].set_title("AQI vs Infection Risk")

sns.barplot(x='Infection_Risk_Level', y='Age', data=df,
            palette='Blues', ax=axes[0,1])
axes[0,1].set_title("Age vs Infection Risk")

sns.barplot(x='Infection_Risk_Level', y='Transmission_Rate', data=df,
            palette='Greens', ax=axes[1,0])
axes[1,0].set_title("Transmission vs Risk")

sns.barplot(x='Infection_Risk_Level', y='Resource_Utilization', data=df,
            palette='Purples', ax=axes[1,1])
axes[1,1].set_title("Resource Utilization vs Risk")

plt.tight_layout()
plt.savefig("plots/plot_08_risk_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
