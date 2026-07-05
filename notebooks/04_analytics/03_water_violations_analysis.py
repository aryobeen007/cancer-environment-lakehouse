# Databricks notebook source
# MAGIC %md
# MAGIC # Analytics 03 - Water Violations vs Cancer Rates
# MAGIC
# MAGIC I'm analyzing the relationship between EPA drinking water health-based violations
# MAGIC and cancer rates across U.S. states. Health-based violations occur when a contaminant
# MAGIC exceeds the Maximum Contaminant Level (MCL) — these are the violations with direct
# MAGIC public health implications.
# MAGIC
# MAGIC Key questions:
# MAGIC - Do states with more health-based water violations have higher cancer rates?
# MAGIC - Is MCL violation count or contaminant diversity more predictive?
# MAGIC - Which states have the worst combined water quality + cancer burden?
# MAGIC - How do water violations correlate with cancer mortality specifically?
# MAGIC
# MAGIC **Source table:** `gold.state_environmental_risk_profile`, `gold.state_water_violations_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Water violations vs cancer rates — state overview

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(avg_cancer_incidence_rate, 1) AS cancer_incidence_rate,
# MAGIC     ROUND(avg_cancer_mortality_rate, 1) AS cancer_mortality_rate,
# MAGIC     total_health_violations_all_years,
# MAGIC     ROUND(avg_health_violations_per_year, 1) AS avg_violations_per_year,
# MAGIC     total_mcl_violations,
# MAGIC     ROUND(avg_distinct_contaminants, 1) AS avg_contaminants,
# MAGIC     ROUND(avg_enforcement_rate_pct, 1) AS enforcement_rate_pct,
# MAGIC     is_high_cancer_incidence
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC WHERE total_health_violations_all_years IS NOT NULL
# MAGIC ORDER BY total_health_violations_all_years DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pearson correlations — water violations vs cancer rates

# COMMAND ----------

from pyspark.sql import functions as F

df_profile = spark.table(gold_table("state_environmental_risk_profile"))

water_metrics = [
    "total_health_violations_all_years",
    "avg_health_violations_per_year",
    "total_mcl_violations",
    "avg_distinct_contaminants",
    "total_tier1_notifications"
]

cancer_metrics = [
    "avg_cancer_incidence_rate",
    "avg_cancer_mortality_rate"
]

print("Pearson Correlations — Water Violations vs Cancer Rates")
print("=" * 65)
for water_col in water_metrics:
    for cancer_col in cancer_metrics:
        corr = df_profile.stat.corr(water_col, cancer_col)
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"{water_col:<40} vs {cancer_col:<30}: r={corr:+.3f} {direction} ({strength})")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Water violations trend over time — national average

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     year,
# MAGIC     SUM(total_health_violations) AS total_violations,
# MAGIC     SUM(mcl_violations) AS mcl_violations,
# MAGIC     COUNT(DISTINCT state_abbr) AS states_reporting,
# MAGIC     ROUND(AVG(distinct_contaminants), 1) AS avg_contaminants_per_state,
# MAGIC     ROUND(AVG(enforcement_rate_pct), 1) AS avg_enforcement_rate
# MAGIC FROM cancer_environment_lakehouse.gold.state_water_violations_summary
# MAGIC GROUP BY year
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. States with worst combined water violations AND cancer burden

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(avg_cancer_mortality_rate, 1) AS cancer_mortality_rate,
# MAGIC     total_health_violations_all_years,
# MAGIC     total_mcl_violations,
# MAGIC     ROUND(avg_distinct_contaminants, 1) AS avg_contaminants,
# MAGIC     total_tier1_notifications,
# MAGIC     ROUND(avg_smoking_pct, 1) AS smoking_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC WHERE total_health_violations_all_years IS NOT NULL
# MAGIC   AND avg_cancer_mortality_rate IS NOT NULL
# MAGIC ORDER BY avg_cancer_mortality_rate DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Violation quintile analysis — cancer rates by violation burden

# COMMAND ----------

from pyspark.sql.window import Window

df_clean = df_profile.filter(
    F.col("total_health_violations_all_years").isNotNull() &
    F.col("avg_cancer_mortality_rate").isNotNull()
)

window = Window.orderBy("total_health_violations_all_years")
df_quintile = (
    df_clean
    .withColumn("row_num", F.row_number().over(window))
    .withColumn("total_rows", F.count("state_abbr").over(Window.partitionBy()))
    .withColumn("violation_quintile",
        F.ceil(F.col("row_num") * 5 / F.col("total_rows")))
    .drop("row_num", "total_rows")
)

display(
    df_quintile
    .groupBy("violation_quintile")
    .agg(
        F.count("state_abbr").alias("state_count"),
        F.round(F.avg("total_health_violations_all_years"), 0).alias("avg_violations"),
        F.round(F.avg("avg_cancer_incidence_rate"), 1).alias("avg_cancer_incidence"),
        F.round(F.avg("avg_cancer_mortality_rate"), 1).alias("avg_cancer_mortality"),
        F.collect_list("state_abbr").alias("states")
    )
    .orderBy("violation_quintile")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key findings summary

# COMMAND ----------

print("""
KEY FINDINGS — Water Violations vs Cancer Rates
===============================================

Note: Correlation does not imply causation. State-level aggregation
masks county-level variation in both water violations and cancer rates.

1. TEXAS LEADS violations (153,507 health-based) but has moderate cancer
   rates — suggesting water treatment compliance and population size effects.

2. HIGH MORTALITY + HIGH VIOLATIONS: States like Oklahoma, Louisiana, and
   Ohio appear in both high violation counts and high cancer mortality rankings,
   suggesting a potential compounding effect worth further investigation.

3. ENFORCEMENT IS HIGH: Average enforcement rates of 95%+ across most states
   indicate violations are being acted upon — but enforcement after the fact
   doesn't undo the exposure that already occurred.

4. CONTAMINANT DIVERSITY matters: States with higher average distinct
   contaminants per year (PA: 12.9, CA: 12.3, TX: 12.0) may face more
   complex multi-contaminant exposure scenarios.

5. TIER 1 NOTIFICATIONS indicate the most acute violations — states with
   high Tier 1 counts faced contamination events serious enough to require
   immediate public notification.
""")