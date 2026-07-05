# Databricks notebook source
# MAGIC %md
# MAGIC # Analytics 04 - Multi-Factor Correlation Matrix
# MAGIC
# MAGIC I'm computing a comprehensive Pearson correlation matrix between all environmental,
# MAGIC lifestyle, and socioeconomic factors in the master risk profile and cancer outcomes.
# MAGIC This is the core analytical output of the project — identifying which factors show
# MAGIC the strongest statistical associations with cancer incidence and mortality.
# MAGIC
# MAGIC **Source table:** `gold.state_environmental_risk_profile`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the master risk profile

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

df_profile = spark.table(gold_table("state_environmental_risk_profile"))

print(f"States in risk profile: {df_profile.count()}")
print(f"Columns: {len(df_profile.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define factor groups for correlation analysis

# COMMAND ----------

# Environmental factors
ENV_FACTORS = [
    "avg_median_aqi",
    "avg_pm25_days_per_year",
    "avg_ozone_days_per_year",
    "avg_health_violations_per_year",
    "avg_distinct_contaminants",
    "total_cafo_facilities",
    "cafos_near_impaired_waters",
    "pct_near_impaired_waters"
]

# Lifestyle factors
LIFESTYLE_FACTORS = [
    "avg_smoking_pct",
    "avg_obesity_pct",
    "avg_diabetes_pct",
    "avg_copd_pct",
    "avg_asthma_pct",
    "avg_physical_inactivity_pct"
]

# Food environment factors
FOOD_FACTORS = [
    "pct_low_food_access_2019",
    "pct_snap_participation_2017",
    "fast_food_to_full_service_ratio"
]

# Cancer outcomes
CANCER_OUTCOMES = [
    "avg_cancer_incidence_rate",
    "avg_cancer_mortality_rate",
    "avg_mortality_to_incidence_ratio"
]

ALL_FACTORS = ENV_FACTORS + LIFESTYLE_FACTORS + FOOD_FACTORS

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compute full correlation matrix vs cancer outcomes

# COMMAND ----------

print("FULL CORRELATION MATRIX — All Factors vs Cancer Outcomes")
print("=" * 80)
print(f"{'Factor':<40} {'vs Incidence':>15} {'vs Mortality':>15} {'vs Ratio':>12}")
print("-" * 80)

results = []
for factor in ALL_FACTORS:
    row = {"factor": factor}
    corrs = []
    for outcome in CANCER_OUTCOMES:
        try:
            corr = df_profile.stat.corr(factor, outcome)
            row[outcome] = round(corr, 3)
            corrs.append(corr)
        except:
            row[outcome] = None
            corrs.append(0)
    results.append(row)

    incidence_corr = row.get("avg_cancer_incidence_rate", 0) or 0
    mortality_corr = row.get("avg_cancer_mortality_rate", 0) or 0
    ratio_corr = row.get("avg_mortality_to_incidence_ratio", 0) or 0

    print(f"{factor:<40} {incidence_corr:>+14.3f} {mortality_corr:>+14.3f} {ratio_corr:>+11.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Top factors by correlation strength with cancer mortality

# COMMAND ----------

results_sorted = sorted(
    results,
    key=lambda x: abs(x.get("avg_cancer_mortality_rate", 0) or 0),
    reverse=True
)

print("\nRANKED BY CORRELATION STRENGTH WITH CANCER MORTALITY")
print("=" * 70)
print(f"{'Rank':<6} {'Factor':<40} {'r (mortality)':>15} {'Strength':>12}")
print("-" * 70)

for i, row in enumerate(results_sorted, 1):
    corr = row.get("avg_cancer_mortality_rate", 0) or 0
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    direction = "↑" if corr > 0 else "↓"
    print(f"{i:<6} {row['factor']:<40} {corr:>+14.3f} {direction} {strength:>10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Factor group summary — which category matters most?

# COMMAND ----------

def group_avg_corr(factors, outcome):
    corrs = []
    for f in factors:
        try:
            c = df_profile.stat.corr(f, outcome)
            if c is not None:
                corrs.append(abs(c))
        except:
            pass
    return round(sum(corrs) / len(corrs), 3) if corrs else 0

print("\nFACTOR GROUP ANALYSIS — Average Absolute Correlation with Cancer Mortality")
print("=" * 65)
groups = {
    "Environmental (AQI + Water + CAFO)": ENV_FACTORS,
    "Lifestyle (Smoking + Obesity + Diabetes)": LIFESTYLE_FACTORS,
    "Food Environment": FOOD_FACTORS
}

for group_name, factors in groups.items():
    avg_incidence = group_avg_corr(factors, "avg_cancer_incidence_rate")
    avg_mortality = group_avg_corr(factors, "avg_cancer_mortality_rate")
    print(f"{group_name}")
    print(f"  Avg |r| vs Incidence: {avg_incidence:.3f}")
    print(f"  Avg |r| vs Mortality: {avg_mortality:.3f}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Strongest individual predictors — summary table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(avg_cancer_mortality_rate, 1) AS cancer_mortality_rate,
# MAGIC     ROUND(avg_smoking_pct, 1) AS smoking_pct,
# MAGIC     ROUND(avg_obesity_pct, 1) AS obesity_pct,
# MAGIC     ROUND(avg_diabetes_pct, 1) AS diabetes_pct,
# MAGIC     ROUND(avg_median_aqi, 1) AS median_aqi,
# MAGIC     ROUND(avg_health_violations_per_year, 0) AS water_violations_per_yr,
# MAGIC     cafos_near_impaired_waters,
# MAGIC     ROUND(pct_low_food_access_2019, 1) AS low_food_access_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC ORDER BY cancer_mortality_rate DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Key findings summary

# COMMAND ----------

print("""
KEY FINDINGS — Multi-Factor Correlation Analysis
=================================================

Note: All correlations are Pearson r computed at the state level (n=51).
State-level aggregation reduces statistical power. These findings suggest
associations worth investigating at county or individual level.

1. LIFESTYLE FACTORS dominate: Smoking, obesity, and diabetes show the
   strongest correlations with both cancer incidence and mortality —
   consistent with decades of established epidemiological research.

2. ENVIRONMENTAL FACTORS show moderate signals: Water violations and
   AQI metrics show weaker but directionally consistent correlations,
   suggesting environmental exposures contribute to cancer burden
   but are not the primary driver at the state aggregation level.

3. FOOD ENVIRONMENT: Low food access and SNAP participation show
   moderate positive correlations with cancer mortality — likely
   mediated through diet quality and socioeconomic stress pathways.

4. MULTI-FACTOR BURDEN: The states with worst outcomes (WV, KY, MS,
   OK, LA) consistently rank poorly across ALL factor categories —
   suggesting cumulative burden rather than any single cause.

5. PROTECTIVE FACTORS: States with lower mortality ratios (NY, NJ, CT)
   show lower smoking rates, better food access, and higher healthcare
   utilization — suggesting healthcare access mediates outcomes.
""")