# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - EPA Air Quality Index (AQI) by County (2000-2022)
# MAGIC
# MAGIC I'm ingesting 23 annual EPA AQI CSV files (2000-2022) into the Bronze layer in a single
# MAGIC Spark read using a wildcard path. This unions all 23 files automatically and adds a
# MAGIC `source_file` column so I can always trace a row back to its original annual file.
# MAGIC
# MAGIC **Source files:** `annual_aqi_by_county_2000.csv` → `annual_aqi_by_county_2022.csv` (23 files)
# MAGIC **Target table:** `cancer_environment_lakehouse.bronze.air_quality`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read all 23 annual files with a wildcard path
# MAGIC
# MAGIC Spark reads all matching files and unions them into a single DataFrame. I use
# MAGIC `input_file_name()` to capture which annual file each row came from, so I can
# MAGIC derive the year later in Silver without relying solely on the Year column.

# COMMAND ----------

from pyspark.sql import functions as F

wildcard_path = f"{LANDING_VOLUME_PATH}/annual_aqi_by_county_*.csv"

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(wildcard_path)
    .withColumn("source_file", F.col("_metadata.file_path"))
)

print(f"Read {df_raw.count():,} rows from {wildcard_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect the schema and sample rows

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

display(df_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Confirm all 23 annual files were read
# MAGIC
# MAGIC I want to verify every file contributed rows - if any file had a schema mismatch
# MAGIC or was empty, it will be missing here.

# COMMAND ----------

display(
    df_raw
    .groupBy("source_file")
    .count()
    .orderBy("source_file")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sanitize column names
# MAGIC
# MAGIC EPA AQI exports use spaces and mixed case in column names. I apply the same
# MAGIC generic sanitizer I used for the cancer incidence notebook.

# COMMAND ----------

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
# MAGIC ## 5. Add ingestion timestamp
# MAGIC
# MAGIC The source_file column is already set from input_file_name() above, so I only
# MAGIC need to add the ingested_at timestamp here.

# COMMAND ----------

df_bronze = df_raw.withColumn("ingested_at", F.current_timestamp())

display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to the Bronze Delta table

# COMMAND ----------

target_table = BRONZE_TABLES["air_quality"]

write_bronze_table(df_bronze, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate the write

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.air_quality;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cancer_environment_lakehouse.bronze.air_quality LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Confirm year coverage across all 23 files

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT year, COUNT(*) AS county_count
# MAGIC FROM cancer_environment_lakehouse.bronze.air_quality
# MAGIC GROUP BY year
# MAGIC ORDER BY year;