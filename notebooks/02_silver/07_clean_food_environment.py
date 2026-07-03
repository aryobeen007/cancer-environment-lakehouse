# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - USDA Food Environment Atlas
# MAGIC
# MAGIC I'm cleaning and transforming the Bronze Food Environment Atlas table into Silver.
# MAGIC The data is already in long format (one row per county × variable_code) which
# MAGIC simplifies the transformation significantly — no unpivoting needed.
# MAGIC
# MAGIC Key decisions I'm making here:
# MAGIC
# MAGIC 1. **Join variable descriptions** — join the VariableList table to get human-readable
# MAGIC    category and variable names for each `variable_code`
# MAGIC 2. **Filter to relevant variables** — keep only food access, food insecurity, store
# MAGIC    availability, and restaurant density variables most relevant to cancer-environment
# MAGIC    analysis
# MAGIC 3. **Standardize FIPS codes** — pad FIPS to 5 characters for county-level joins
# MAGIC 4. **Add state abbreviation** — extract from the 2-char state field already present
# MAGIC
# MAGIC **Source tables:** `bronze.food_environment_state_county`, `bronze.food_environment_variable_list`
# MAGIC **Target table:** `silver.food_environment`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Bronze tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df_atlas = spark.table(bronze_table("food_environment_state_county"))
df_vars = spark.table(bronze_table("food_environment_variable_list"))

print(f"Atlas rows: {df_atlas.count():,}")
print(f"Variable list rows: {df_vars.count():,}")

# COMMAND ----------

display(df_atlas.limit(5))

# COMMAND ----------

display(df_vars.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect variable list schema and categories

# COMMAND ----------

df_vars.printSchema()

# COMMAND ----------

display(
    df_vars.groupBy("category_name")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sanitize variable list column names and join to atlas

# COMMAND ----------

import re

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

# Sanitize variable list columns
for col in df_vars.columns:
    new_name = sanitize_column_name(col)
    if new_name != col:
        df_vars = df_vars.withColumnRenamed(col, new_name)

df_vars.printSchema()

# COMMAND ----------

# Join variable descriptions to atlas data
df_joined = df_atlas.join(
    F.broadcast(df_vars),
    on="variable_code",
    how="left"
)

print(f"Joined rows: {df_joined.count():,}")
display(df_joined.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Standardize FIPS code and cast value

# COMMAND ----------

df_silver = (
    df_joined
    # Pad FIPS to 5 characters (some may be 4 chars missing leading zero)
    .withColumn("county_fips",
        F.lpad(F.col("fips").cast("string"), 5, "0"))

    # Extract state FIPS (first 2 chars of county FIPS)
    .withColumn("state_fips",
        F.col("county_fips").substr(1, 2))

    # Cast value to double
    .withColumn("value", F.col("value").cast(DoubleType()))

    # Rename columns for clarity
    .withColumnRenamed("state", "state_abbr")
    .withColumnRenamed("county", "county_name")

    # Drop original fips and metadata columns
    .drop("fips", "ingested_at", "source_file")
)

df_silver.printSchema()

# COMMAND ----------

display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check null values and variable coverage

# COMMAND ----------

null_values = df_silver.filter(F.col("value").isNull()).count()
total = df_silver.count()
print(f"Null values: {null_values:,} of {total:,} ({null_values/total*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Silver

# COMMAND ----------

target_table = "cancer_environment_lakehouse.silver.food_environment"

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT county_fips) AS counties,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     COUNT(DISTINCT variable_code) AS variables,
# MAGIC     SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_values
# MAGIC FROM cancer_environment_lakehouse.silver.food_environment;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Preview — food insecurity rate by state

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     ROUND(AVG(value), 1) AS avg_food_insecurity_pct
# MAGIC FROM cancer_environment_lakehouse.silver.food_environment
# MAGIC WHERE variable_code = 'FOODINSEC_15_17'
# MAGIC   AND value IS NOT NULL
# MAGIC GROUP BY state_abbr
# MAGIC ORDER BY avg_food_insecurity_pct DESC
# MAGIC LIMIT 15;