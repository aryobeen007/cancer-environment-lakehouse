# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - EPA Air Quality Index (AQI) by County (2000-2022)
# MAGIC
# MAGIC I'm cleaning and transforming the Bronze AQI table into a production-ready Silver table.
# MAGIC Key decisions I'm making here:
# MAGIC
# MAGIC 1. **Rename `90th_percentile_aqi`** — column names starting with a number cause issues
# MAGIC    in some SQL contexts, so I rename it to `percentile_90_aqi`
# MAGIC 2. **Extract year from source_file** — as a cross-check against the `year` column
# MAGIC 3. **Remove non-US entries** — Canada, Mexico, Virgin Islands, Puerto Rico can't join
# MAGIC    to state-level cancer data
# MAGIC 4. **Fix DC casing** — EPA uses "District Of Columbia", lookup uses "District of Columbia"
# MAGIC 5. **Add state abbreviation** — for joining to state-level cancer datasets
# MAGIC 6. **Flag low monitoring coverage** — counties with fewer than 50 days are unreliable
# MAGIC 7. **Add AQI category** — classifies median AQI into EPA standard buckets
# MAGIC
# MAGIC **Source table:** `cancer_environment_lakehouse.bronze.air_quality`
# MAGIC **Target table:** `cancer_environment_lakehouse.silver.air_quality`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read from Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

df_bronze = spark.table(bronze_table("air_quality"))

print(f"Bronze row count: {df_bronze.count():,}")
display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Rename, cast, and extract year from filename

# COMMAND ----------

df_silver = (
    df_bronze
    .withColumnRenamed("90th_percentile_aqi", "percentile_90_aqi")
    .withColumn("year_from_file",
        F.regexp_extract(F.col("source_file"), r"(\d{4})\.csv$", 1).cast(IntegerType()))
    .withColumn("year", F.col("year").cast(IntegerType()))
    .withColumn("days_with_aqi", F.col("days_with_aqi").cast(IntegerType()))
    .withColumn("good_days", F.col("good_days").cast(IntegerType()))
    .withColumn("moderate_days", F.col("moderate_days").cast(IntegerType()))
    .withColumn("unhealthy_for_sensitive_groups_days", F.col("unhealthy_for_sensitive_groups_days").cast(IntegerType()))
    .withColumn("unhealthy_days", F.col("unhealthy_days").cast(IntegerType()))
    .withColumn("very_unhealthy_days", F.col("very_unhealthy_days").cast(IntegerType()))
    .withColumn("hazardous_days", F.col("hazardous_days").cast(IntegerType()))
    .withColumn("max_aqi", F.col("max_aqi").cast(IntegerType()))
    .withColumn("percentile_90_aqi", F.col("percentile_90_aqi").cast(IntegerType()))
    .withColumn("median_aqi", F.col("median_aqi").cast(IntegerType()))
    .withColumn("days_co", F.col("days_co").cast(IntegerType()))
    .withColumn("days_no2", F.col("days_no2").cast(IntegerType()))
    .withColumn("days_ozone", F.col("days_ozone").cast(IntegerType()))
    .withColumn("days_pm25", F.col("days_pm25").cast(IntegerType()))
    .withColumn("days_pm10", F.col("days_pm10").cast(IntegerType()))
    .drop("source_file", "ingested_at")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify year matches year_from_file

# COMMAND ----------

mismatches = df_silver.filter(F.col("year") != F.col("year_from_file"))
print(f"Year mismatches between column and filename: {mismatches.count()}")
df_silver = df_silver.drop("year_from_file")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Remove non-US entries and fix DC casing

# COMMAND ----------

non_us = ["Canada", "Country Of Mexico", "Virgin Islands", "Puerto Rico"]
df_silver = df_silver.filter(~F.col("state").isin(non_us))

df_silver = df_silver.withColumn(
    "state",
    F.when(F.col("state") == "District Of Columbia", "District of Columbia")
    .otherwise(F.col("state"))
)

print(f"Rows after removing non-US entries: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add state abbreviation lookup

# COMMAND ----------

state_abbrev = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
}

state_abbrev_rows = [(k, v) for k, v in state_abbrev.items()]
df_abbrev = spark.createDataFrame(state_abbrev_rows, ["state", "state_abbr"])

df_silver = df_silver.join(F.broadcast(df_abbrev), on="state", how="left")

unmatched = df_silver.filter(F.col("state_abbr").isNull()).select("state").distinct()
print(f"Unmatched states: {unmatched.count()}")
display(unmatched)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Flag low-coverage counties and add AQI category

# COMMAND ----------

df_silver = (
    df_silver
    .withColumn("low_coverage",
        F.when(F.col("days_with_aqi") < 50, True).otherwise(False))
    .withColumn("aqi_category",
        F.when(F.col("median_aqi") <= 50, "Good")
        .when(F.col("median_aqi") <= 100, "Moderate")
        .when(F.col("median_aqi") <= 150, "Unhealthy for Sensitive Groups")
        .when(F.col("median_aqi") <= 200, "Unhealthy")
        .when(F.col("median_aqi") <= 300, "Very Unhealthy")
        .otherwise("Hazardous"))
)

low_coverage_count = df_silver.filter(F.col("low_coverage") == True).count()
total = df_silver.count()
print(f"Low coverage counties: {low_coverage_count:,} of {total:,} ({low_coverage_count/total*100:.1f}%)")

display(
    df_silver.groupBy("aqi_category")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Final column ordering and write to Silver

# COMMAND ----------

df_silver = df_silver.select(
    "state",
    "state_abbr",
    "county",
    "year",
    "days_with_aqi",
    "good_days",
    "moderate_days",
    "unhealthy_for_sensitive_groups_days",
    "unhealthy_days",
    "very_unhealthy_days",
    "hazardous_days",
    "max_aqi",
    "percentile_90_aqi",
    "median_aqi",
    "aqi_category",
    "days_co",
    "days_no2",
    "days_ozone",
    "days_pm25",
    "days_pm10",
    "low_coverage"
)

print(f"Final Silver row count: {df_silver.count():,}")
df_silver.printSchema()

# COMMAND ----------

target_table = silver_table("air_quality")

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data quality summary

# COMMAND ----------

data_quality_summary(df_silver, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Validate — coverage and stats

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state) AS states,
# MAGIC     COUNT(DISTINCT county) AS counties,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     ROUND(AVG(median_aqi), 1) AS avg_median_aqi,
# MAGIC     SUM(CASE WHEN low_coverage THEN 1 ELSE 0 END) AS low_coverage_rows
# MAGIC FROM cancer_environment_lakehouse.silver.air_quality;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Top 10 states by average median AQI (worst air quality)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state,
# MAGIC     state_abbr,
# MAGIC     ROUND(AVG(median_aqi), 1) AS avg_median_aqi,
# MAGIC     ROUND(AVG(max_aqi), 1) AS avg_max_aqi,
# MAGIC     COUNT(DISTINCT county) AS counties_monitored
# MAGIC FROM cancer_environment_lakehouse.silver.air_quality
# MAGIC WHERE low_coverage = false
# MAGIC GROUP BY state, state_abbr
# MAGIC ORDER BY avg_median_aqi DESC
# MAGIC LIMIT 10;