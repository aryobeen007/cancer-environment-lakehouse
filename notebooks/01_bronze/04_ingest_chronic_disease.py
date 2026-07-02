# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - CDC Chronic Disease Indicators (All States)
# MAGIC
# MAGIC I'm ingesting the CDC Chronic Disease Indicators dataset into the Bronze layer.
# MAGIC This is a long-format dataset where each row represents one indicator measurement
# MAGIC for one state, year, and stratification combination (Sex, Age, Race, Overall, etc.).
# MAGIC
# MAGIC Key characteristics of this dataset:
# MAGIC - 34 columns, CDC long-format structure (Topic + Question + DataValue)
# MAGIC - Covers multiple chronic disease topics: Asthma, Cancer, Cardiovascular, Diabetes,
# MAGIC   Smoking, Obesity, and more
# MAGIC - Three stratification layers allow breakdowns by Sex, Age, and Race
# MAGIC - DataValueFootnoteSymbol = '*' indicates suppressed or unavailable data
# MAGIC - Geolocation column contains WKT POINT strings for mapping
# MAGIC
# MAGIC **Source file:** `cdc_chronic_disease_indicators_all_states.csv`
# MAGIC **Target table:** `cancer_environment_lakehouse.bronze.chronic_disease`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the raw CSV

# COMMAND ----------

source_file = "cdc_chronic_disease_indicators_all_states.csv"
source_path = f"{LANDING_VOLUME_PATH}/{source_file}"

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(source_path)
)

print(f"Read {df_raw.count():,} rows from {source_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect the schema and sample rows

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

display(df_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sanitize column names
# MAGIC
# MAGIC Column names are CamelCase with no spaces, so the sanitizer just lowercases them.
# MAGIC No special character issues expected here unlike the EPA/CDC WONDER exports.

# COMMAND ----------

import re

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip()
    # Insert underscore between camelcase boundaries
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

renamed_columns = {col: sanitize_column_name(col) for col in df_raw.columns}
print(renamed_columns)

for old_name, new_name in renamed_columns.items():
    df_raw = df_raw.withColumnRenamed(old_name, new_name)

df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick topic coverage check
# MAGIC
# MAGIC I want to confirm which chronic disease topics are in this dataset before writing
# MAGIC to Bronze. This shapes which topics I'll filter for in Silver.

# COMMAND ----------

from pyspark.sql import functions as F

display(
    df_raw.groupBy("topic")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add ingestion metadata

# COMMAND ----------

df_bronze = add_ingestion_metadata(df_raw, source_file)

display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to the Bronze Delta table

# COMMAND ----------

target_table = BRONZE_TABLES["chronic_disease"]

write_bronze_table(df_bronze, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate the write

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.chronic_disease;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cancer_environment_lakehouse.bronze.chronic_disease LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Year and state coverage check

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     MIN(year_start) AS earliest_year,
# MAGIC     MAX(year_end) AS latest_year,
# MAGIC     COUNT(DISTINCT location_abbr) AS state_count,
# MAGIC     COUNT(DISTINCT topic) AS topic_count,
# MAGIC     COUNT(DISTINCT question) AS question_count
# MAGIC FROM cancer_environment_lakehouse.bronze.chronic_disease;