# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - USDA Census of Agriculture (Livestock Operations)
# MAGIC
# MAGIC I'm ingesting the USDA Census of Agriculture QuickStats file into the Bronze layer.
# MAGIC This is a space-delimited .txt file (~2.24 GB) with many columns covering crop,
# MAGIC livestock, and demographic data across U.S. counties.
# MAGIC
# MAGIC **Source file:** `qs_census2022.txt`
# MAGIC **Target table:** `cancer_environment_lakehouse.bronze.usda_livestock`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the raw text and detect actual delimiter

# COMMAND ----------

source_file = "qs_census2022.txt"
source_path = f"{LANDING_VOLUME_PATH}/{source_file}"

# Read first line as raw text to see the actual delimiter
first_line = spark.read.text(source_path).limit(1).collect()[0][0]
print("First 500 characters of header row:")
print(first_line[:500])
print()

# Count potential delimiters
print(f"Comma count:     {first_line.count(',')}")
print(f"Pipe count:      {first_line.count('|')}")
print(f"Tab count:       {first_line.count(chr(9))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read with correct delimiter
# MAGIC
# MAGIC Based on the counts above, the delimiter with the highest count is the correct one.
# MAGIC USDA QuickStats files are typically comma-delimited with quoted fields.

# COMMAND ----------

from pyspark.sql import functions as F
import re

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", ",")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", "true")
    .csv(source_path)
)

print(f"Column count: {len(df_raw.columns)}")
print(f"Columns: {df_raw.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sanitize column names
# MAGIC
# MAGIC Each column gets sanitized individually — lowercase, spaces/hyphens to underscores,
# MAGIC special characters removed. Column names are truncated to 100 chars max to avoid
# MAGIC the 255-char Delta limit.

# COMMAND ----------

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    # Truncate to 100 chars to stay well within Delta's 255-char limit
    return name[:100]

# Sanitize and deduplicate column names
seen = {}
new_columns = []
for col in df_raw.columns:
    sanitized = sanitize_column_name(col)
    if sanitized in seen:
        seen[sanitized] += 1
        sanitized = f"{sanitized}_{seen[sanitized]}"
    else:
        seen[sanitized] = 0
    new_columns.append((col, sanitized))

print("Column mappings:")
for old, new in new_columns:
    print(f"  {old!r:40} -> {new!r}")

for old, new in new_columns:
    df_raw = df_raw.withColumnRenamed(old, new)

print(f"\nFinal column count: {len(df_raw.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample rows and row count

# COMMAND ----------

print(f"Row count: {df_raw.count():,}")

# COMMAND ----------

display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add ingestion metadata and write to Bronze

# COMMAND ----------

df_bronze = add_ingestion_metadata(df_raw, source_file)
target_table = BRONZE_TABLES["usda_livestock"]
write_bronze_table(df_bronze, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate the write

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.usda_livestock;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cancer_environment_lakehouse.bronze.usda_livestock LIMIT 10;