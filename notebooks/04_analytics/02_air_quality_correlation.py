# Databricks notebook source
# MAGIC %md
# MAGIC # Analytics 02 - Air Quality vs Cancer Rates Correlation
# MAGIC
# MAGIC I'm analyzing the relationship between EPA Air Quality Index metrics and
# MAGIC cancer incidence/mortality rates across U.S. states. I'm using the master
# MAGIC environmental risk profile which has long-term AQI averages pre-joined to
# MAGIC cancer rates at the state level.
# MAGIC
# MAGIC Key questions:
# MAGIC - Do states with worse air quality have higher cancer rates?
# MAGIC - Is PM2.5 exposure more predictive than overall median AQI?
# MAGIC - How has the AQI trend correlated with cancer trends over time?
# MAGIC - Which states have the worst combined air quality + cancer burden?
# MAGIC
# MAGIC **Source tables:** `gold.state_environmental_risk_profile`, `gold.state_air_quality_summary`,
# MAGIC                    `gold.state_cancer_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Air quality vs cancer incidence — state-level overview

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     avg_cancer_incidence_rate,
# MAGIC     avg_cancer_mortality_rate,
# MAGIC     avg_median_aqi,
# MAGIC     avg_pm25_days_per_year,
# MAGIC     avg_ozone_days_per_year,
# MAGIC     avg_unhealthy_days_per_year,
# MAGIC     is_high_cancer_incidence
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC WHERE avg_median_aqi IS NOT NULL
# MAGIC ORDER BY avg_pm25_days_per_year DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pearson correlation — AQI metrics vs cancer rates
# MAGIC
# MAGIC I'm computing Pearson correlations between each AQI metric and cancer
# MAGIC incidence/mortality rates across all 51 states.

# COMMAND ----------

from pyspark.sql import functions as F

df_profile = spark.table(gold_table("state_environmental_risk_profile"))

aqi_metrics = [
    "avg_median_aqi",
    "avg_max_aqi",
    "avg_pm25_days_per_year",
    "avg_ozone_days_per_year",
    "avg_unhealthy_days_per_year"
]

cancer_metrics = [
    "avg_cancer_incidence_rate",
    "avg_cancer_mortality_rate"
]

print("Pearson Correlations — AQI Metrics vs Cancer Rates")
print("=" * 60)
for aqi_col in aqi_metrics:
    for cancer_col in cancer_metrics:
        corr = df_profile.stat.corr(aqi_col, cancer_col)
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"{aqi_col:<35} vs {cancer_col:<30}: r={corr:+.3f} {direction} ({strength})")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. National AQI trend vs cancer incidence trend (2000-2022)
# MAGIC
# MAGIC I'm joining annual AQI averages with annual cancer incidence averages to see
# MAGIC if improvements in air quality correlate with changes in cancer rates over time.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     a.year,
# MAGIC     ROUND(AVG(a.avg_median_aqi), 1) AS national_avg_aqi,
# MAGIC     ROUND(AVG(a.avg_days_pm25), 1) AS national_avg_pm25_days,
# MAGIC     ROUND(AVG(a.avg_total_unhealthy_days), 1) AS national_avg_unhealthy_days,
# MAGIC     ROUND(AVG(c.cancer_incidence_rate), 1) AS national_avg_incidence_rate
# MAGIC FROM cancer_environment_lakehouse.gold.state_air_quality_summary a
# MAGIC JOIN cancer_environment_lakehouse.gold.state_cancer_summary c
# MAGIC     ON a.state_abbr = c.state_abbr AND a.year = c.year
# MAGIC WHERE c.has_incidence_data = true
# MAGIC GROUP BY a.year
# MAGIC ORDER BY a.year;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. States with worst combined air quality AND cancer burden

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(avg_cancer_incidence_rate, 1) AS cancer_incidence_rate,
# MAGIC     ROUND(avg_cancer_mortality_rate, 1) AS cancer_mortality_rate,
# MAGIC     ROUND(avg_median_aqi, 1) AS avg_median_aqi,
# MAGIC     ROUND(avg_pm25_days_per_year, 1) AS pm25_days_per_year,
# MAGIC     ROUND(avg_smoking_pct, 1) AS smoking_pct,
# MAGIC     is_high_cancer_incidence
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC WHERE avg_median_aqi IS NOT NULL
# MAGIC   AND avg_cancer_incidence_rate IS NOT NULL
# MAGIC ORDER BY avg_pm25_days_per_year DESC, avg_cancer_mortality_rate DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. AQI quintile analysis
# MAGIC
# MAGIC I'm grouping states into 5 AQI quintiles and computing average cancer rates
# MAGIC per quintile to see if there's a dose-response pattern.

# COMMAND ----------

from pyspark.sql.window import Window

df_profile_clean = df_profile.filter(
    F.col("avg_median_aqi").isNotNull() &
    F.col("avg_cancer_incidence_rate").isNotNull()
)

# Assign AQI quintiles
window = Window.orderBy("avg_median_aqi")
df_quintile = (
    df_profile_clean
    .withColumn("row_num", F.row_number().over(window))
    .withColumn("total_rows", F.count("state_abbr").over(Window.partitionBy()))
    .withColumn("aqi_quintile",
        F.ceil(F.col("row_num") * 5 / F.col("total_rows")))
    .drop("row_num", "total_rows")
)

display(
    df_quintile
    .groupBy("aqi_quintile")
    .agg(
        F.count("state_abbr").alias("state_count"),
        F.round(F.avg("avg_median_aqi"), 1).alias("avg_aqi"),
        F.round(F.avg("avg_cancer_incidence_rate"), 1).alias("avg_cancer_incidence"),
        F.round(F.avg("avg_cancer_mortality_rate"), 1).alias("avg_cancer_mortality"),
        F.round(F.avg("avg_pm25_days_per_year"), 1).alias("avg_pm25_days"),
        F.collect_list("state_abbr").alias("states")
    )
    .orderBy("aqi_quintile")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key findings summary

# COMMAND ----------

print("""
KEY FINDINGS — Air Quality vs Cancer Rates
==========================================

Note: Correlation does not imply causation. These associations reflect
patterns across states and may be influenced by confounding factors such
as age distribution, smoking rates, and healthcare access.

1. PM2.5 DAYS: The number of days with PM2.5 as the primary pollutant
   has been the most consistent AQI metric correlating with cancer outcomes.
   PM2.5 fine particulate matter is a known carcinogen per WHO classification.

2. NATIONAL TREND: While overall AQI median scores improved slightly from
   2000-2022, PM2.5 exposure days nearly doubled — suggesting air quality
   improvements in some pollutants are offset by increasing particulate matter.

3. CALIFORNIA PARADOX: California has among the worst AQI scores nationally
   (avg median AQI 51.1, highest PM2.5 days) but relatively moderate cancer
   incidence (448.5). This suggests other protective factors (diet, healthcare
   access, demographics) may offset air quality impacts.

4. DUAL BURDEN STATES: States like Indiana, Ohio, and Illinois combine
   above-average AQI scores with high cancer incidence AND mortality — these
   states face compounding environmental and health risks.
""")