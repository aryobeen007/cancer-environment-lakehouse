# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Workspace Config
# MAGIC
# MAGIC I'm defining all the shared paths, catalog/schema names, and utility functions I'll reuse
# MAGIC across every Bronze, Silver, and Gold notebook in this project. Centralizing these here
# MAGIC means I only update one place if a path or naming convention changes.
# MAGIC
# MAGIC I run this notebook with `%run` at the top of every other notebook in the project.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog & Schema Constants

# COMMAND ----------

CATALOG = "cancer_environment_lakehouse"

SCHEMA_RAW = "raw"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
SCHEMA_GOLD = "gold"

LANDING_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/landing"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source File Names
# MAGIC
# MAGIC I'm mapping each of my 9 source datasets to its filename in the landing volume.
# MAGIC I'll update these to match my actual filenames once I upload the CSVs.

# COMMAND ----------

SOURCE_FILES = {
    "cancer_incidence": "cancer_incidence_by_state.csv",
    "cancer_mortality": "cancer_mortality_by_state.csv",
    "air_quality": "epa_aqi_*.csv",  # 23 annual files, handled with a wildcard read
    "water_violations": "sdwis_water_violations.csv",
    "water_systems": "sdwis_water_systems.csv",
    "food_environment": "usda_food_environment_atlas.csv",
    "chronic_disease": "cdc_brfss_chronic_disease.csv",
    "livestock_operations": "usda_census_agriculture_livestock.csv",
    "cafo_facilities": "epa_echo_cafo_facilities.csv",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Table Names
# MAGIC
# MAGIC Fully qualified three-level names (`catalog.schema.table`) for every Bronze table.

# COMMAND ----------

BRONZE_TABLES = {
    "cancer_incidence": f"{CATALOG}.{SCHEMA_BRONZE}.cancer_incidence",
    "cancer_mortality": f"{CATALOG}.{SCHEMA_BRONZE}.cancer_mortality",
    "air_quality": f"{CATALOG}.{SCHEMA_BRONZE}.air_quality",
    "water_violations": f"{CATALOG}.{SCHEMA_BRONZE}.water_violations",
    "water_systems": f"{CATALOG}.{SCHEMA_BRONZE}.water_systems",
    "food_environment": f"{CATALOG}.{SCHEMA_BRONZE}.food_environment",
    "chronic_disease": f"{CATALOG}.{SCHEMA_BRONZE}.chronic_disease",
    "livestock_operations": f"{CATALOG}.{SCHEMA_BRONZE}.livestock_operations",
    "cafo_facilities": f"{CATALOG}.{SCHEMA_BRONZE}.cafo_facilities",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver & Gold Table Name Helpers
# MAGIC
# MAGIC Rather than hardcoding every Silver/Gold table name, I use small helper functions
# MAGIC so notebooks downstream just pass a short key.

# COMMAND ----------

def silver_table(name: str) -> str:
    """Returns the fully qualified Silver table name for a given dataset key."""
    return f"{CATALOG}.{SCHEMA_SILVER}.{name}"


def gold_table(name: str) -> str:
    """Returns the fully qualified Gold table name for a given dataset key."""
    return f"{CATALOG}.{SCHEMA_GOLD}.{name}"


def bronze_table(name: str) -> str:
    """Returns the fully qualified Bronze table name for a given dataset key."""
    return f"{CATALOG}.{SCHEMA_BRONZE}.{name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Utility Functions

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


def add_ingestion_metadata(df: DataFrame, source_file: str) -> DataFrame:
    """
    Adds standard metadata columns to a Bronze DataFrame before I write it to Delta.
    I add this to every Bronze table so I can always trace a row back to its source file
    and ingestion time.
    """
    return (
        df.withColumn("ingested_at", F.current_timestamp())
        .withColumn("source_file", F.lit(source_file))
    )


def write_bronze_table(df: DataFrame, table_name: str, mode: str = "overwrite") -> None:
    """
    Writes a DataFrame to a Bronze Delta table and prints a row count summary.
    I default to overwrite since Bronze ingestion in this project is a full refresh,
    not incremental.
    """
    df.write.format("delta").mode(mode).saveAsTable(table_name)
    row_count = df.count()
    print(f"Wrote {row_count:,} rows to {table_name}")


def data_quality_summary(df: DataFrame, table_name: str) -> None:
    """
    Prints a quick data quality summary for a DataFrame - row count, column count,
    and null counts per column. I run this at the end of every Silver notebook.
    """
    print(f"\n--- Data Quality Summary: {table_name} ---")
    print(f"Row count: {df.count():,}")
    print(f"Column count: {len(df.columns)}")
    print("\nNull counts per column:")
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    )
    null_counts.show(truncate=False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Broadcast-Friendly Dimension Table Names
# MAGIC
# MAGIC Per my performance plan, `dim_state` (51 rows) and `dim_year` (47 rows) get broadcast
# MAGIC in every join. I reference them here so every notebook uses the same names.

# COMMAND ----------

DIM_STATE_TABLE = gold_table("dim_state")
DIM_YEAR_TABLE = gold_table("dim_year")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm Config Loaded

# COMMAND ----------

print("Workspace config loaded successfully.")
print(f"Catalog: {CATALOG}")
print(f"Landing volume path: {LANDING_VOLUME_PATH}")
print(f"Bronze tables defined: {len(BRONZE_TABLES)}")