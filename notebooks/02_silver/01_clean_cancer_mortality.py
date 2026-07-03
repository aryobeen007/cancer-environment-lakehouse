# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - Cancer Mortality by State
# MAGIC
# MAGIC I'm cleaning and transforming the Bronze cancer mortality table into a production-ready
# MAGIC Silver table. The key decisions I'm making here:
# MAGIC
# MAGIC 1. **Filter out Total rollup rows** — CDC WONDER includes a pre-aggregated "Total" row
# MAGIC    per state. I drop these since I can always recompute aggregations myself, and keeping
# MAGIC    them would double-count data in joins.
# MAGIC 2. **Cast all columns to correct types** — Bronze used inferSchema which got most types
# MAGIC    right, but I'm being explicit here for Silver.
# MAGIC 3. **Standardize column names** — rename `notes` suppression flag to `is_suppressed`
# MAGIC    boolean so downstream queries don't need to know CDC's convention.
# MAGIC 4. **Add FIPS code** — join state_code to a FIPS lookup so I can join to county-level
# MAGIC    datasets later.
# MAGIC
# MAGIC **Source table:** `cancer_environment_lakehouse.bronze.cancer_mortality`
# MAGIC **Target table:** `cancer_environment_lakehouse.silver.cancer_mortality`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read from Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType

df_bronze = spark.table(bronze_table("cancer_mortality"))

print(f"Bronze row count: {df_bronze.count():,}")
display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect what's in the notes column
# MAGIC
# MAGIC Before filtering, I want to confirm exactly what values appear in `notes`
# MAGIC so I don't accidentally drop rows I need.

# COMMAND ----------

display(
    df_bronze.groupBy("notes")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter out Total and grand total rows
# MAGIC
# MAGIC I keep only year-level rows where notes IS NULL — these are the actual annual
# MAGIC state-level measurements. Rows where notes = 'Total' are pre-aggregated rollups
# MAGIC per state, and rows with methodology footnote text in notes are CDC metadata rows.

# COMMAND ----------

df_filtered = df_bronze.filter(
    F.col("notes").isNull() &
    F.col("year").isNotNull() &
    F.col("state").isNotNull()
)

print(f"Rows after filtering: {df_filtered.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cast columns to correct types and rename for clarity

# COMMAND ----------

df_silver = (
    df_filtered
    # Explicit type casting
    .withColumn("year", F.col("year").cast(IntegerType()))
    .withColumn("state_code", F.col("state_code").cast(IntegerType()))
    .withColumn("deaths", F.col("deaths").cast(IntegerType()))
    .withColumn("population", F.col("population").cast(IntegerType()))
    .withColumn("age_adjusted_rate", F.col("age_adjusted_rate").cast(DoubleType()))

    # Rename columns for clarity
    .withColumnRenamed("state", "state_name")
    .withColumnRenamed("state_code", "state_fips")

    # Add suppression flag — True if data was suppressed by CDC
    .withColumn("is_suppressed", F.lit(False))

    # Drop columns not needed in Silver
    .drop("notes", "year_code", "ingested_at", "source_file")
)

df_silver.printSchema()

# COMMAND ----------

display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add state abbreviation lookup
# MAGIC
# MAGIC I'm adding a state abbreviation column so I can join to other datasets that
# MAGIC use abbreviations (e.g. CDC BRFSS uses 'AL', 'AK', etc.) rather than full names.

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

# Create a mapping DataFrame and join
state_abbrev_rows = [(k, v) for k, v in state_abbrev.items()]
df_abbrev = spark.createDataFrame(state_abbrev_rows, ["state_name", "state_abbr"])

df_silver = df_silver.join(
    F.broadcast(df_abbrev),
    on="state_name",
    how="left"
)

# Check for any states that didn't match
unmatched = df_silver.filter(F.col("state_abbr").isNull()).select("state_name").distinct()
print(f"Unmatched states (no abbreviation found): {unmatched.count()}")
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
    "deaths",
    "population",
    "age_adjusted_rate",
    "is_suppressed"
)

print(f"Final Silver row count: {df_silver.count():,}")
df_silver.printSchema()

# COMMAND ----------

target_table = silver_table("cancer_mortality")

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data quality summary

# COMMAND ----------

data_quality_summary(df_silver, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validate — state and year coverage

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
# MAGIC FROM cancer_environment_lakehouse.silver.cancer_mortality;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quick preview — top 10 states by average mortality rate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_name,
# MAGIC     state_abbr,
# MAGIC     ROUND(AVG(age_adjusted_rate), 1) AS avg_mortality_rate,
# MAGIC     SUM(deaths) AS total_deaths
# MAGIC FROM cancer_environment_lakehouse.silver.cancer_mortality
# MAGIC GROUP BY state_name, state_abbr
# MAGIC ORDER BY avg_mortality_rate DESC
# MAGIC LIMIT 10;