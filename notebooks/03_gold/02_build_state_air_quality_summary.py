# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Air Quality Summary
# MAGIC
# MAGIC I'm aggregating county-level AQI data up to state-year level for the Gold layer.
# MAGIC I exclude low-coverage counties (fewer than 50 monitoring days) from state averages
# MAGIC to avoid rural monitoring gaps skewing the results.
# MAGIC
# MAGIC Key metrics I'm computing:
# MAGIC - Average median AQI per state per year
# MAGIC - Total unhealthy days (unhealthy + very unhealthy + hazardous) per state per year
# MAGIC - PM2.5 days as a cancer-relevant pollutant metric
# MAGIC - County monitoring coverage count
# MAGIC
# MAGIC **Source table:** `silver.air_quality`
# MAGIC **Target table:** `gold.state_air_quality_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver table

# COMMAND ----------

from pyspark.sql import functions as F

df_aqi = spark.table(silver_table("air_quality"))

print(f"Silver AQI rows: {df_aqi.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate to state-year level
# MAGIC
# MAGIC I exclude low_coverage counties and compute weighted averages where possible.

# COMMAND ----------

df_gold = (
    df_aqi
    .filter(F.col("low_coverage") == False)
    .groupBy("state", "state_abbr", "year")
    .agg(
        # AQI metrics
        F.round(F.avg("median_aqi"), 1).alias("avg_median_aqi"),
        F.round(F.avg("max_aqi"), 1).alias("avg_max_aqi"),
        F.round(F.avg("percentile_90_aqi"), 1).alias("avg_90th_pct_aqi"),

        # Unhealthy days — sum across all counties then average per county
        F.round(F.avg("unhealthy_days"), 1).alias("avg_unhealthy_days"),
        F.round(F.avg("very_unhealthy_days"), 1).alias("avg_very_unhealthy_days"),
        F.round(F.avg("hazardous_days"), 1).alias("avg_hazardous_days"),
        F.round(F.avg("good_days"), 1).alias("avg_good_days"),

        # Total unhealthy days combined
        F.round(F.avg(
            F.col("unhealthy_days") +
            F.col("very_unhealthy_days") +
            F.col("hazardous_days")
        ), 1).alias("avg_total_unhealthy_days"),

        # Pollutant-specific days — cancer-relevant
        F.round(F.avg("days_pm25"), 1).alias("avg_days_pm25"),
        F.round(F.avg("days_ozone"), 1).alias("avg_days_ozone"),
        F.round(F.avg("days_no2"), 1).alias("avg_days_no2"),
        F.round(F.avg("days_co"), 1).alias("avg_days_co"),

        # Monitoring coverage
        F.count("county").alias("counties_monitored"),
        F.sum("days_with_aqi").alias("total_monitoring_days")
    )
    .orderBy("state_abbr", "year")
)

print(f"Gold rows: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

display(df_gold.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write to Gold

# COMMAND ----------

target_table = gold_table("state_air_quality_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     ROUND(AVG(avg_median_aqi), 1) AS overall_avg_aqi,
# MAGIC     ROUND(AVG(avg_days_pm25), 1) AS overall_avg_pm25_days
# MAGIC FROM cancer_environment_lakehouse.gold.state_air_quality_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Trend — national average AQI by year (2000-2022)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     year,
# MAGIC     ROUND(AVG(avg_median_aqi), 1) AS national_avg_aqi,
# MAGIC     ROUND(AVG(avg_total_unhealthy_days), 1) AS national_avg_unhealthy_days,
# MAGIC     ROUND(AVG(avg_days_pm25), 1) AS national_avg_pm25_days,
# MAGIC     SUM(counties_monitored) AS total_counties_monitored
# MAGIC FROM cancer_environment_lakehouse.gold.state_air_quality_summary
# MAGIC GROUP BY year
# MAGIC ORDER BY year;