# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Environmental Risk Profile (Master Join Table)
# MAGIC
# MAGIC I'm building the master analytical table for this project by joining all 6 Gold
# MAGIC summary tables into a single wide-format state-level table. This table is the
# MAGIC primary input for:
# MAGIC - All analytics and correlation notebooks
# MAGIC - Both ML models (regression and classifier)
# MAGIC - All three Tableau dashboards
# MAGIC
# MAGIC Join strategy:
# MAGIC - Anchor: `state_cancer_summary` (incidence years 1999-2022)
# MAGIC - All other tables join on `state_abbr`
# MAGIC - Time-varying tables (AQI, water violations) are aggregated to state-level
# MAGIC   averages before joining since they don't share the same year grain as cancer
# MAGIC - Static tables (CAFO, food environment, lifestyle) join directly
# MAGIC
# MAGIC **Source tables:** All 6 Gold summary tables
# MAGIC **Target table:** `gold.state_environmental_risk_profile`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read all Gold summary tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

df_cancer = spark.table(gold_table("state_cancer_summary"))
df_aqi = spark.table(gold_table("state_air_quality_summary"))
df_water = spark.table(gold_table("state_water_violations_summary"))
df_cafo = spark.table(gold_table("state_cafo_summary"))
df_lifestyle = spark.table(gold_table("state_lifestyle_summary"))
df_food = spark.table(gold_table("state_food_environment_summary"))

print(f"Cancer summary:    {df_cancer.count():,} rows")
print(f"AQI summary:       {df_aqi.count():,} rows")
print(f"Water violations:  {df_water.count():,} rows")
print(f"CAFO summary:      {df_cafo.count():,} rows")
print(f"Lifestyle summary: {df_lifestyle.count():,} rows")
print(f"Food environment:  {df_food.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate time-varying tables to state-level averages
# MAGIC
# MAGIC AQI and water violations vary by year. I compute long-term averages per state
# MAGIC so they can join to the cancer incidence data which spans 1999-2022.

# COMMAND ----------

# AQI — average across all years (2000-2022)
df_aqi_avg = (
    df_aqi
    .groupBy("state_abbr")
    .agg(
        F.round(F.avg("avg_median_aqi"), 1).alias("avg_median_aqi"),
        F.round(F.avg("avg_max_aqi"), 1).alias("avg_max_aqi"),
        F.round(F.avg("avg_total_unhealthy_days"), 1).alias("avg_unhealthy_days_per_year"),
        F.round(F.avg("avg_days_pm25"), 1).alias("avg_pm25_days_per_year"),
        F.round(F.avg("avg_days_ozone"), 1).alias("avg_ozone_days_per_year"),
        F.round(F.avg("counties_monitored"), 0).cast(IntegerType()).alias("avg_counties_monitored")
    )
)

print(f"AQI state averages: {df_aqi_avg.count():,} states")

# COMMAND ----------

# Water violations — total and averages across all years
df_water_avg = (
    df_water
    .groupBy("state_abbr")
    .agg(
        F.sum("total_health_violations").alias("total_health_violations_all_years"),
        F.round(F.avg("total_health_violations"), 1).alias("avg_health_violations_per_year"),
        F.sum("mcl_violations").alias("total_mcl_violations"),
        F.round(F.avg("distinct_contaminants"), 1).alias("avg_distinct_contaminants"),
        F.round(F.avg("enforcement_rate_pct"), 1).alias("avg_enforcement_rate_pct"),
        F.sum("tier1_notifications").alias("total_tier1_notifications")
    )
)

print(f"Water violations state averages: {df_water_avg.count():,} states")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Aggregate cancer summary to state-level averages
# MAGIC
# MAGIC Cancer data varies by year. I compute state averages across all available years
# MAGIC for the risk profile, plus most recent year metrics.

# COMMAND ----------

df_cancer_avg = (
    df_cancer
    .filter(F.col("has_incidence_data") == True)
    .groupBy("state_abbr", "state_name", "state_fips")
    .agg(
        F.round(F.avg("cancer_incidence_rate"), 1).alias("avg_cancer_incidence_rate"),
        F.round(F.avg("cancer_mortality_rate"), 1).alias("avg_cancer_mortality_rate"),
        F.round(F.avg("mortality_to_incidence_ratio"), 1).alias("avg_mortality_to_incidence_ratio"),
        F.sum("cancer_new_cases").alias("total_new_cases"),
        F.sum("cancer_deaths").alias("total_deaths"),
        F.count("year").alias("years_of_incidence_data")
    )
)

print(f"Cancer state averages: {df_cancer_avg.count():,} states")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregate lifestyle to state-level averages

# COMMAND ----------

df_lifestyle_avg = (
    df_lifestyle
    .groupBy("state_abbr")
    .agg(
        F.round(F.avg("smoking_prevalence_pct"), 1).alias("avg_smoking_pct"),
        F.round(F.avg("obesity_prevalence_pct"), 1).alias("avg_obesity_pct"),
        F.round(F.avg("diabetes_prevalence_pct"), 1).alias("avg_diabetes_pct"),
        F.round(F.avg("asthma_prevalence_pct"), 1).alias("avg_asthma_pct"),
        F.round(F.avg("copd_prevalence_pct"), 1).alias("avg_copd_pct"),
        F.round(F.avg("physical_inactivity_pct"), 1).alias("avg_physical_inactivity_pct")
    )
)

print(f"Lifestyle state averages: {df_lifestyle_avg.count():,} states")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build master join

# COMMAND ----------

df_risk_profile = (
    df_cancer_avg

    # Join AQI
    .join(df_aqi_avg, on="state_abbr", how="left")

    # Join water violations
    .join(df_water_avg, on="state_abbr", how="left")

    # Join CAFO (select key columns only)
    .join(
        df_cafo.select(
            "state_abbr",
            "total_cafo_facilities",
            "cafos_near_impaired_waters",
            "pct_near_impaired_waters",
            "cattle_facilities",
            "dairy_facilities",
            "poultry_facilities"
        ),
        on="state_abbr", how="left"
    )

    # Join lifestyle
    .join(df_lifestyle_avg, on="state_abbr", how="left")

    # Join food environment (select key columns only)
    .join(
        df_food.select(
            "state_abbr",
            "pct_low_food_access_2019",
            "pct_snap_participation_2017",
            "grocery_stores_per_1000_2016",
            "fast_food_restaurants_per_1000_2016",
            "fast_food_to_full_service_ratio",
            "pct_obese_adults_2017",
            "pct_diabetes_adults_2015"
        ),
        on="state_abbr", how="left"
    )

    # Add high risk flag — states in top quartile for cancer incidence rate
    .withColumn("is_high_cancer_incidence",
        F.when(F.col("avg_cancer_incidence_rate") >= 500, True).otherwise(False))

    .orderBy("state_abbr")
)

print(f"Final risk profile rows: {df_risk_profile.count():,}")
print(f"Final column count: {len(df_risk_profile.columns)}")
df_risk_profile.printSchema()

# COMMAND ----------

display(df_risk_profile.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Gold

# COMMAND ----------

target_table = gold_table("state_environmental_risk_profile")

df_risk_profile.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_states,
# MAGIC     SUM(CASE WHEN is_high_cancer_incidence THEN 1 ELSE 0 END) AS high_incidence_states,
# MAGIC     ROUND(AVG(avg_cancer_incidence_rate), 1) AS national_avg_incidence,
# MAGIC     ROUND(AVG(avg_cancer_mortality_rate), 1) AS national_avg_mortality,
# MAGIC     ROUND(AVG(avg_median_aqi), 1) AS national_avg_aqi,
# MAGIC     ROUND(AVG(avg_smoking_pct), 1) AS national_avg_smoking,
# MAGIC     ROUND(AVG(avg_obesity_pct), 1) AS national_avg_obesity
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Full risk profile — top 15 states by cancer incidence rate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     avg_cancer_incidence_rate,
# MAGIC     avg_cancer_mortality_rate,
# MAGIC     avg_mortality_to_incidence_ratio,
# MAGIC     avg_median_aqi,
# MAGIC     avg_smoking_pct,
# MAGIC     avg_obesity_pct,
# MAGIC     total_health_violations_all_years,
# MAGIC     cafos_near_impaired_waters,
# MAGIC     pct_low_food_access_2019,
# MAGIC     is_high_cancer_incidence
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC ORDER BY avg_cancer_incidence_rate DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Correlation preview — smoking vs cancer incidence

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     avg_smoking_pct,
# MAGIC     avg_cancer_incidence_rate,
# MAGIC     avg_cancer_mortality_rate,
# MAGIC     avg_obesity_pct,
# MAGIC     avg_median_aqi
# MAGIC FROM cancer_environment_lakehouse.gold.state_environmental_risk_profile
# MAGIC WHERE avg_smoking_pct IS NOT NULL
# MAGIC   AND avg_cancer_incidence_rate IS NOT NULL
# MAGIC ORDER BY avg_smoking_pct DESC
# MAGIC LIMIT 15;