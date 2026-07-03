# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - USDA Census of Agriculture (Livestock Operations)
# MAGIC
# MAGIC The Bronze table for this dataset has a parsing issue — the file is tab-delimited
# MAGIC but was read with a comma delimiter, resulting in all columns being crammed into one.
# MAGIC
# MAGIC I'm fixing this by reading the raw file directly from the landing volume with the
# MAGIC correct tab delimiter, applying Silver-quality transformations, and writing straight
# MAGIC to the Silver table — bypassing Bronze entirely for this dataset.
# MAGIC
# MAGIC The file has 39 columns covering crop, livestock, and demographic data across
# MAGIC U.S. counties from the 2022 USDA Census of Agriculture.
# MAGIC
# MAGIC For this project I'm filtering to livestock and animal production rows only
# MAGIC (SECTOR_DESC = 'ANIMALS & PRODUCTS') since those are the rows relevant to
# MAGIC CAFO and agricultural impact analysis.
# MAGIC
# MAGIC **Source file:** `/Volumes/cancer_environment_lakehouse/raw/landing/qs_census2022.txt`
# MAGIC **Target table:** `cancer_environment_lakehouse.silver.usda_livestock`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read raw file with correct tab delimiter

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, LongType
import re

source_path = f"{LANDING_VOLUME_PATH}/qs_census2022.txt"

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")  # All string first — we'll cast explicitly
    .option("sep", "\t")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", "true")
    .csv(source_path)
)

print(f"Raw row count: {df_raw.count():,}")
print(f"Column count: {len(df_raw.columns)}")
print(f"Columns: {df_raw.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sanitize column names

# COMMAND ----------

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name[:100]

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

for old, new in new_columns:
    df_raw = df_raw.withColumnRenamed(old, new)

print("Sanitized column names:")
print(df_raw.columns)

# COMMAND ----------

display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check sector distribution and filter to Animals & Products

# COMMAND ----------

display(
    df_raw.groupBy("sector_desc")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

df_filtered = df_raw.filter(F.col("sector_desc") == "ANIMALS & PRODUCTS")
print(f"Animals & Products rows: {df_filtered.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Select and cast key columns

# COMMAND ----------

df_silver = (
    df_filtered
    .select(
        "source_desc",
        "sector_desc",
        "group_desc",
        "commodity_desc",
        "class_desc",
        "prodn_practice_desc",
        "util_practice_desc",
        "statisticcat_desc",
        "unit_desc",
        "short_desc",
        "domain_desc",
        "domaincat_desc",
        "agg_level_desc",
        F.col("state_ansi").alias("state_fips"),
        F.col("state_alpha").alias("state_abbr"),
        F.col("state_name"),
        F.col("county_ansi").alias("county_fips"),
        F.col("county_name"),
        F.col("year").cast(IntegerType()).alias("year"),
        F.col("value").alias("value_raw"),
    )

    # Clean value column — USDA uses ' (D)' for suppressed and ' (Z)' for zero
    .withColumn("is_suppressed",
        F.when(F.trim(F.col("value_raw")).isin("(D)", "(Z)", "(H)", "(L)", "(X)"), True)
        .otherwise(False))

    .withColumn("value",
        F.when(F.col("is_suppressed"), F.lit(None).cast(DoubleType()))
        .otherwise(
            F.regexp_replace(F.col("value_raw"), ",", "").cast(DoubleType())
        ))

    .drop("value_raw")
)

print(f"Silver row count: {df_silver.count():,}")
df_silver.printSchema()

# COMMAND ----------

display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check suppression rate and commodity coverage

# COMMAND ----------

suppressed = df_silver.filter(F.col("is_suppressed")).count()
total = df_silver.count()
print(f"Suppressed values: {suppressed:,} of {total:,} ({suppressed/total*100:.1f}%)")

# COMMAND ----------

display(
    df_silver.groupBy("commodity_desc")
    .count()
    .orderBy(F.desc("count"))
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Silver

# COMMAND ----------

target_table = "cancer_environment_lakehouse.silver.usda_livestock"

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     COUNT(DISTINCT commodity_desc) AS commodities,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     SUM(CASE WHEN is_suppressed THEN 1 ELSE 0 END) AS suppressed_rows
# MAGIC FROM cancer_environment_lakehouse.silver.usda_livestock;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Preview — inventory counts by state for key livestock types

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     commodity_desc,
# MAGIC     statisticcat_desc,
# MAGIC     ROUND(SUM(value), 0) AS total_value,
# MAGIC     unit_desc
# MAGIC FROM cancer_environment_lakehouse.silver.usda_livestock
# MAGIC WHERE commodity_desc IN ('HOGS', 'CATTLE', 'POULTRY', 'BROILERS')
# MAGIC   AND statisticcat_desc = 'INVENTORY'
# MAGIC   AND agg_level_desc = 'STATE'
# MAGIC   AND is_suppressed = false
# MAGIC   AND value IS NOT NULL
# MAGIC GROUP BY state_abbr, commodity_desc, statisticcat_desc, unit_desc
# MAGIC ORDER BY total_value DESC
# MAGIC LIMIT 15;