# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - Cancer Incidence by State
# MAGIC
# MAGIC I'm ingesting the CDC WONDER cancer incidence dataset (1999-2022) into the Bronze layer.
# MAGIC This follows the same pattern I proved out with the cancer mortality notebook, including
# MAGIC the column name sanitization step I had to add after hitting a Delta error there.
# MAGIC
# MAGIC **Source file:** `cdc_wonder_cancer_incidence_by_state_1999_2022.csv`
# MAGIC **Target table:** `cancer_environment_lakehouse.bronze.cancer_incidence`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the raw CSV

# COMMAND ----------

source_file = "cdc_wonder_cancer_incidence_by_state_1999_2022.csv"
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
# MAGIC Delta doesn't allow spaces or special characters in column names. I'm renaming to
# MAGIC snake_case here in Bronze, since this is a structural fix rather than a data cleaning
# MAGIC decision. CDC WONDER exports tend to use the same column naming convention across
# MAGIC datasets, so I expect most of these renames to match what I used for cancer mortality.
# MAGIC
# MAGIC I print the schema first so I can confirm the actual column names before mapping them -
# MAGIC incidence data may have a couple of extra columns mortality didn't (e.g. a rate type
# MAGIC or cancer site breakdown).

# COMMAND ----------

# Print current columns so I can confirm names before renaming
print(df_raw.columns)

# COMMAND ----------

# Generic sanitizer as a safety net: lowercases, replaces spaces/hyphens with underscores,
# strips any other invalid characters. I still keep an explicit mapping below for clarity
# and so I have full control over the final names.
import re

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
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
# MAGIC ## 4. Add ingestion metadata

# COMMAND ----------

df_bronze = add_ingestion_metadata(df_raw, source_file)

display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write to the Bronze Delta table

# COMMAND ----------

target_table = BRONZE_TABLES["cancer_incidence"]

write_bronze_table(df_bronze, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate the write

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.cancer_incidence;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cancer_environment_lakehouse.bronze.cancer_incidence LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quick sanity check on key columns

# COMMAND ----------

display(
    spark.table(target_table)
    .select("*")
    .summary()
)