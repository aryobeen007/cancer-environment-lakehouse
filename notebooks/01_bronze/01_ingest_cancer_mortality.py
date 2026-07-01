# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - Cancer Mortality by State
# MAGIC
# MAGIC I'm ingesting the CDC WONDER cancer mortality dataset (2018-2023) into the Bronze layer.
# MAGIC This is my smallest dataset, so I'm using it to prove out the ingestion pattern I'll
# MAGIC reuse for the rest of the Bronze notebooks.
# MAGIC
# MAGIC **Source file:** `cdc_wonder_cancer_mortality_by_state_2018_2023.csv`
# MAGIC **Target table:** `cancer_environment_lakehouse.bronze.cancer_mortality`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the raw CSV
# MAGIC
# MAGIC I'm reading with `header=True` and `inferSchema=True` for now. I'll lock down explicit
# MAGIC types in the Silver notebook - Bronze should stay close to the raw source.

# COMMAND ----------

source_file = "cdc_wonder_cancer_mortality_by_state_2018_2023.csv"
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
# MAGIC
# MAGIC Before I write anything, I want to confirm the columns and types came through as expected.

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

display(df_raw.limit(10))

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3a. Sanitize column names
# MAGIC
# MAGIC Delta doesn't allow spaces or special characters in column names. I'm renaming columns
# MAGIC to snake_case here, in Bronze, since this is a structural fix rather than a data
# MAGIC cleaning decision — I want every downstream notebook working with clean column names.

# COMMAND ----------

column_renames = {
    "State Code": "state_code",
    "Year Code": "year_code",
    "Age-Adjusted Rate": "age_adjusted_rate",
    "Notes": "notes",
    "State": "state",
    "Year": "year",
    "Deaths": "deaths",
    "Population": "population",
}

for old_name, new_name in column_renames.items():
    df_raw = df_raw.withColumnRenamed(old_name, new_name)

df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Add ingestion metadata
# MAGIC
# MAGIC I stamp every Bronze row with `ingested_at` and `source_file` so I can trace it back later.

# COMMAND ----------

df_bronze = add_ingestion_metadata(df_raw, source_file)

display(df_bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write to the Bronze Delta table

# COMMAND ----------

target_table = BRONZE_TABLES["cancer_mortality"]

write_bronze_table(df_bronze, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate the write

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.cancer_mortality;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cancer_environment_lakehouse.bronze.cancer_mortality LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Quick sanity check on key columns
# MAGIC
# MAGIC I want to confirm state coverage looks complete (should be ~51 for all states + DC)
# MAGIC before I move on to Silver cleaning.

# COMMAND ----------

display(
    spark.table(target_table)
    .select("*")
    .summary()
)