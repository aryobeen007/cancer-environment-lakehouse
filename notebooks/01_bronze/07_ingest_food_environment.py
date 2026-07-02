# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - USDA Food Environment Atlas
# MAGIC
# MAGIC I'm ingesting two Food Environment Atlas files into the Bronze layer:
# MAGIC - `StateAndCountyData.csv` — the wide-format atlas data with hundreds of food
# MAGIC   environment indicators per county (food access, store counts, restaurant counts,
# MAGIC   food insecurity rates, etc.)
# MAGIC - `VariableList.csv` — the data dictionary mapping variable codes to descriptions,
# MAGIC   which I'll use during Silver unpivoting to label each indicator correctly
# MAGIC
# MAGIC Note: The StateAndCountyData file is in wide format (one row per county, hundreds
# MAGIC of columns). I'll unpivot it to long format in the Silver notebook.
# MAGIC
# MAGIC **Source files:** `StateAndCountyData.csv`, `VariableList.csv`
# MAGIC **Target tables:** `cancer_environment_lakehouse.bronze.food_environment_state_county`
# MAGIC                    `cancer_environment_lakehouse.bronze.food_environment_variable_list`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import re
from pyspark.sql import functions as F

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest StateAndCountyData.csv
# MAGIC
# MAGIC This is the main wide-format atlas file. I'm reading it as-is into Bronze —
# MAGIC the unpivot to long format happens in Silver, not here.

# COMMAND ----------

source_file_main = "StateAndCountyData.csv"
source_path_main = f"{LANDING_VOLUME_PATH}/{source_file_main}"

df_main = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(source_path_main)
)

print(f"Read {df_main.count():,} rows from {source_file_main}")
print(f"Column count: {len(df_main.columns)}")

# COMMAND ----------

df_main.printSchema()

# COMMAND ----------

display(df_main.limit(5))

# COMMAND ----------

# Sanitize column names
renamed_main = {col: sanitize_column_name(col) for col in df_main.columns}
print(f"Sample column renames: {dict(list(renamed_main.items())[:10])}")

for old, new in renamed_main.items():
    df_main = df_main.withColumnRenamed(old, new)

# Add metadata and write
df_main_bronze = add_ingestion_metadata(df_main, source_file_main)
target_main = BRONZE_TABLES["food_environment_state_county"]
write_bronze_table(df_main_bronze, target_main)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest VariableList.csv
# MAGIC
# MAGIC This is the data dictionary — small reference file mapping variable codes
# MAGIC to their full descriptions and categories. I'll join this to the unpivoted
# MAGIC Silver table to get human-readable indicator names.

# COMMAND ----------

source_file_vars = "VariableList.csv"
source_path_vars = f"{LANDING_VOLUME_PATH}/{source_file_vars}"

df_vars = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(source_path_vars)
)

print(f"Read {df_vars.count():,} rows from {source_file_vars}")

# COMMAND ----------

df_vars.printSchema()

# COMMAND ----------

display(df_vars.limit(10))

# COMMAND ----------

# Sanitize column names
renamed_vars = {col: sanitize_column_name(col) for col in df_vars.columns}
print(renamed_vars)

for old, new in renamed_vars.items():
    df_vars = df_vars.withColumnRenamed(old, new)

# Add metadata and write
df_vars_bronze = add_ingestion_metadata(df_vars, source_file_vars)
target_vars = BRONZE_TABLES["food_environment_variable_list"]
write_bronze_table(df_vars_bronze, target_vars)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate both tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'food_environment_state_county' AS table_name, COUNT(*) AS row_count,
# MAGIC        COUNT(DISTINCT state) AS state_count
# MAGIC FROM cancer_environment_lakehouse.bronze.food_environment_state_county
# MAGIC UNION ALL
# MAGIC SELECT 'food_environment_variable_list', COUNT(*), NULL
# MAGIC FROM cancer_environment_lakehouse.bronze.food_environment_variable_list;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check column count on the wide-format table
# MAGIC
# MAGIC The Food Environment Atlas is famously wide — I want to confirm how many
# MAGIC indicator columns I'll need to unpivot in Silver.

# COMMAND ----------

wide_col_count = len(spark.table(target_main).columns)
print(f"food_environment_state_county column count: {wide_col_count}")
print("This confirms how many columns I'll need to unpivot in Silver.")

# COMMAND ----------

display(spark.table("cancer_environment_lakehouse.bronze.food_environment_state_county").limit(5))