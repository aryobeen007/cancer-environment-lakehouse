# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - Cancer Incidence by State
# MAGIC
# MAGIC I'm cleaning and transforming the Bronze cancer incidence table into a production-ready
# MAGIC Silver table. This follows the same pattern as cancer mortality Silver, with a few
# MAGIC differences specific to the incidence dataset:
# MAGIC
# MAGIC - Column names used `states` and `states_code` (plural) instead of `state`/`state_code`
# MAGIC - Spans 1999–2022 (24 years vs 6 years for mortality)
# MAGIC - Uses `count` (new cancer cases) instead of `deaths`
# MAGIC - Age-adjusted rate scale is different (~450–550 per 100k for incidence vs ~150 for mortality)
# MAGIC
# MAGIC **Source table:** `cancer_environment_lakehouse.bronze.cancer_incidence`
# MAGIC **Target table:** `cancer_environment_lakehouse.silver.cancer_incidence`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read from Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DoubleType

df_bronze = spark.table(bronze_table("cancer_incidence"))

print(f"Bronze row count: {df_bronze.count():,}")
display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect notes distribution

# COMMAND ----------

display(
    df_bronze.groupBy("notes")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter out Total and metadata rows
# MAGIC
# MAGIC Same logic as mortality — keep only year-level rows where notes IS NULL,
# MAGIC year IS NOT NULL, and states IS NOT NULL.

# COMMAND ----------

df_filtered = df_bronze.filter(
    F.col("notes").isNull() &
    F.col("year").isNotNull() &
    F.col("states").isNotNull()
)

print(f"Rows after filtering: {df_filtered.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cast columns, rename for consistency with mortality Silver table

# COMMAND ----------

df_silver = (
    df_filtered
    # Rename to match mortality Silver naming convention
    .withColumnRenamed("states", "state_name")
    .withColumnRenamed("states_code", "state_fips")
    .withColumnRenamed("count", "new_cases")

    # Explicit type casting
    .withColumn("year", F.col("year").cast(IntegerType()))
    .withColumn("state_fips", F.col("state_fips").cast(IntegerType()))
    .withColumn("new_cases", F.col("new_cases").cast(IntegerType()))
    .withColumn("population", F.col("population").cast(LongType()))
    .withColumn("age_adjusted_rate", F.col("age_adjusted_rate").cast(DoubleType()))

    # Add suppression flag
    .withColumn("is_suppressed", F.lit(False))

    # Drop unneeded columns
    .drop("notes", "year_code", "ingested_at", "source_file")
)

df_silver.printSchema()

# COMMAND ----------

display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add state abbreviation lookup
# MAGIC
# MAGIC Same lookup I used in cancer mortality Silver.

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
df_abbrev = spark.createDataFrame(state_abbrev_rows, ["state_name", "state_abbr"])

df_silver = df_silver.join(
    F.broadcast(df_abbrev),
    on="state_name",
    how="left"
)

unmatched = df_silver.filter(F.col("state_abbr").isNull()).select("state_name").distinct()
print(f"Unmatched states: {unmatched.count()}")
display(unmatched)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final column ordering and write to Silver

# COMMAND ----------

df_silver = df_silver.select(
    "state_name",
    "state_abbr",
    "state_fips",
    "year",
    "new_cases",
    "population",
    "age_adjusted_rate",
    "is_suppressed"
)

print(f"Final Silver row count: {df_silver.count():,}")

# COMMAND ----------

target_table = silver_table("cancer_incidence")

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data quality summary

# COMMAND ----------

data_quality_summary(df_silver, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validate — coverage and stats

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_name) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     ROUND(AVG(age_adjusted_rate), 2) AS avg_age_adjusted_rate,
# MAGIC     ROUND(MIN(age_adjusted_rate), 2) AS min_rate,
# MAGIC     ROUND(MAX(age_adjusted_rate), 2) AS max_rate
# MAGIC FROM cancer_environment_lakehouse.silver.cancer_incidence;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quick preview — top 10 states by average incidence rate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_name,
# MAGIC     state_abbr,
# MAGIC     ROUND(AVG(age_adjusted_rate), 1) AS avg_incidence_rate,
# MAGIC     SUM(new_cases) AS total_new_cases
# MAGIC FROM cancer_environment_lakehouse.silver.cancer_incidence
# MAGIC GROUP BY state_name, state_abbr
# MAGIC ORDER BY avg_incidence_rate DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Cross-check — do incidence and mortality rankings align?
# MAGIC
# MAGIC I want to see if the states with highest incidence also have highest mortality.
# MAGIC A mismatch could indicate differences in healthcare access or cancer type distribution.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     i.state_abbr,
# MAGIC     ROUND(AVG(i.age_adjusted_rate), 1) AS avg_incidence_rate,
# MAGIC     ROUND(AVG(m.age_adjusted_rate), 1) AS avg_mortality_rate,
# MAGIC     ROUND(AVG(m.age_adjusted_rate) / AVG(i.age_adjusted_rate) * 100, 1) AS mortality_to_incidence_pct
# MAGIC FROM cancer_environment_lakehouse.silver.cancer_incidence i
# MAGIC JOIN cancer_environment_lakehouse.silver.cancer_mortality m
# MAGIC     ON i.state_abbr = m.state_abbr
# MAGIC GROUP BY i.state_abbr
# MAGIC ORDER BY avg_incidence_rate DESC
# MAGIC LIMIT 15;