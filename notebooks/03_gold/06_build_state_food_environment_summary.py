# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Food Environment Summary
# MAGIC
# MAGIC I'm aggregating county-level food environment indicators up to state level.
# MAGIC The Food Environment Atlas has 304 variables — I select the most analytically
# MAGIC relevant ones for cancer-environment analysis:
# MAGIC
# MAGIC - Food access indicators (% population with low access to stores)
# MAGIC - Food assistance (SNAP participation, school lunch)
# MAGIC - Store availability (grocery stores, supercenters per 1000 pop)
# MAGIC - Restaurant density (fast food vs full service ratio)
# MAGIC - Health outcomes (diabetes and obesity rates)
# MAGIC
# MAGIC Note: Some county-level values in the atlas are negative due to zero-population
# MAGIC denominators — I replace these with null before aggregating.
# MAGIC
# MAGIC **Source table:** `silver.food_environment`
# MAGIC **Target table:** `gold.state_food_environment_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver table

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df_food = spark.table("cancer_environment_lakehouse.silver.food_environment")
print(f"Silver food environment rows: {df_food.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define key variables to aggregate

# COMMAND ----------

KEY_VARIABLES = {
    # Food Access (% population with low access)
    "PCT_LACCESS_POP15": "pct_low_food_access_2015",
    "PCT_LACCESS_POP19": "pct_low_food_access_2019",
    "PCT_LACCESS_LOWI15": "pct_low_income_low_access_2015",
    "PCT_LACCESS_LOWI19": "pct_low_income_low_access_2019",

    # Food Assistance
    "PCT_SNAP17": "pct_snap_participation_2017",
    "PCT_NSLP17": "pct_school_lunch_2017",

    # Store Availability (per 1,000 population)
    "GROCPTH16": "grocery_stores_per_1000_2016",
    "SUPERCPTH16": "supercenters_per_1000_2016",
    "CONVSPTH16": "convenience_stores_per_1000_2016",
    "SPECSPTH16": "specialty_stores_per_1000_2016",

    # Restaurant Availability
    "FFRPTH16": "fast_food_restaurants_per_1000_2016",
    "FSRPTH16": "full_service_restaurants_per_1000_2016",

    # Health outcomes
    "PCT_DIABETES_ADULTS15": "pct_diabetes_adults_2015",
    "PCT_OBESE_ADULTS17": "pct_obese_adults_2017",
}

print(f"Variables to aggregate: {len(KEY_VARIABLES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter to key variables, remove negative values, and aggregate to state level
# MAGIC
# MAGIC Negative values in the atlas are data artifacts from zero-population county
# MAGIC denominators — I replace them with null before averaging up to state level.

# COMMAND ----------

df_filtered = (
    df_food
    .filter(F.col("variable_code").isin(list(KEY_VARIABLES.keys())))
    # Remove negative values — these are data artifacts, not real measurements
    .withColumn("value",
        F.when(F.col("value") < 0, F.lit(None).cast(DoubleType()))
        .otherwise(F.col("value")))
    .filter(F.col("value").isNotNull())
)

print(f"Filtered rows after removing nulls and negatives: {df_filtered.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregate from county to state — average of valid county values

# COMMAND ----------

df_state = (
    df_filtered
    .groupBy("state_abbr", "variable_code")
    .agg(F.round(F.avg("value"), 2).alias("state_avg_value"))
)

# Pivot to wide format — one column per variable
df_pivoted = (
    df_state
    .groupBy("state_abbr")
    .pivot("variable_code", list(KEY_VARIABLES.keys()))
    .agg(F.first("state_avg_value"))
)

# Rename columns to readable names
for code, name in KEY_VARIABLES.items():
    if code in df_pivoted.columns:
        df_pivoted = df_pivoted.withColumnRenamed(code, name)
    else:
        df_pivoted = df_pivoted.withColumn(name, F.lit(None).cast(DoubleType()))

print(f"Pivoted rows: {df_pivoted.count():,}")
df_pivoted.printSchema()

# COMMAND ----------

display(df_pivoted.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add fast food to full service ratio

# COMMAND ----------

df_gold = (
    df_pivoted
    .withColumn(
        "fast_food_to_full_service_ratio",
        F.when(
            F.col("full_service_restaurants_per_1000_2016").isNotNull() &
            (F.col("full_service_restaurants_per_1000_2016") > 0),
            F.round(
                F.col("fast_food_restaurants_per_1000_2016") /
                F.col("full_service_restaurants_per_1000_2016"), 2
            )
        ).otherwise(F.lit(None).cast(DoubleType()))
    )
    .orderBy("state_abbr")
)

print(f"Final Gold rows: {df_gold.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Gold

# COMMAND ----------

target_table = gold_table("state_food_environment_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     ROUND(AVG(pct_low_food_access_2019), 1) AS national_avg_low_access_pct,
# MAGIC     ROUND(AVG(pct_snap_participation_2017), 1) AS national_avg_snap_pct,
# MAGIC     ROUND(AVG(fast_food_to_full_service_ratio), 2) AS national_avg_ff_ratio,
# MAGIC     ROUND(AVG(pct_obese_adults_2017), 1) AS national_avg_obesity_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_food_environment_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Top 10 states by low food access

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     ROUND(pct_low_food_access_2019, 1) AS low_food_access_pct,
# MAGIC     ROUND(pct_snap_participation_2017, 1) AS snap_participation_pct,
# MAGIC     ROUND(fast_food_to_full_service_ratio, 2) AS ff_to_fs_ratio,
# MAGIC     ROUND(pct_obese_adults_2017, 1) AS obesity_pct_2017,
# MAGIC     ROUND(pct_diabetes_adults_2015, 1) AS diabetes_pct_2015
# MAGIC FROM cancer_environment_lakehouse.gold.state_food_environment_summary
# MAGIC ORDER BY low_food_access_pct DESC
# MAGIC LIMIT 10;