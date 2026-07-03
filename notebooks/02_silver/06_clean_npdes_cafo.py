# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - NPDES / CAFO Facilities
# MAGIC
# MAGIC I'm building a clean CAFO (Concentrated Animal Feeding Operation) Silver table by
# MAGIC joining three Bronze NPDES tables:
# MAGIC
# MAGIC - **`icis_facilities`** — facility location, state, coordinates, impaired waters flag
# MAGIC - **`icis_permits`** — permit type, status, active/expired dates
# MAGIC - **`npdes_naics`** — NAICS codes identifying agricultural/animal production facilities
# MAGIC
# MAGIC CAFO identification strategy:
# MAGIC - NAICS codes starting with '112' = Animal Production and Aquaculture
# MAGIC - These include hog farms (112320), dairy cattle (112120), broiler chickens (112310),
# MAGIC   cattle feedlots (112112), turkeys (112210), and other livestock operations
# MAGIC
# MAGIC Key analytical columns:
# MAGIC - `impaired_waters = '303(D) Listed'` — facility near an impaired waterway
# MAGIC - `state_code` — for joining to cancer and water quality tables
# MAGIC - `geocode_latitude/longitude` — for spatial analysis
# MAGIC
# MAGIC **Source tables:** `bronze.icis_facilities`, `bronze.icis_permits`, `bronze.npdes_naics`
# MAGIC **Target table:** `silver.cafo_facilities`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Bronze tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df_facilities = spark.table(bronze_table("icis_facilities"))
df_permits = spark.table(bronze_table("icis_permits"))
df_naics = spark.table(bronze_table("npdes_naics"))

print(f"Facilities: {df_facilities.count():,}")
print(f"Permits: {df_permits.count():,}")
print(f"NAICS: {df_naics.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Identify CAFO facilities via NAICS codes
# MAGIC
# MAGIC I filter NAICS to animal production codes (112xxx) then join to facilities
# MAGIC via the NPDES permit number to get the list of CAFO permit IDs.

# COMMAND ----------

# Filter to animal production NAICS codes
df_cafo_naics = df_naics.filter(F.col("naics_code").startswith("112"))

print(f"NAICS records for animal production: {df_cafo_naics.count():,}")

# Get distinct NPDES IDs that are CAFOs
cafo_npdes_ids = df_cafo_naics.select("npdes_id").distinct()
print(f"Distinct CAFO permit IDs: {cafo_npdes_ids.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter facilities to CAFOs only and select key columns

# COMMAND ----------

df_cafo_facilities = (
    df_facilities
    # Keep only facilities with CAFO NAICS codes
    .join(cafo_npdes_ids, on="npdes_id", how="inner")

    # Select key columns
    .select(
        "npdes_id",
        "icis_facility_interest_id",
        "facility_name",
        "state_code",
        "city",
        "county_code",
        "zip",
        "geocode_latitude",
        F.col("geocode_longitude").cast(DoubleType()).alias("geocode_longitude"),
        "impaired_waters",
        "location_address"
    )

    # Standardize impaired waters flag to boolean
    .withColumn("near_impaired_waters",
        F.when(F.col("impaired_waters") == "303(D) Listed", True)
        .otherwise(False))

    # Clean up latitude
    .withColumn("geocode_latitude",
        F.col("geocode_latitude").cast(DoubleType()))

    .drop("impaired_waters")
)

print(f"CAFO facilities: {df_cafo_facilities.count():,}")
display(df_cafo_facilities.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Join permit information
# MAGIC
# MAGIC I join the permits table to get permit status, type, and active date range.
# MAGIC I keep only active permits (permit_status_code = 'EFF') to focus on currently
# MAGIC operating CAFOs.

# COMMAND ----------

df_permits_clean = (
    df_permits
    .select(
        F.col("external_permit_nmbr").alias("npdes_id"),
        "permit_type_code",
        "permit_status_code",
        "major_minor_status_flag",
        "original_issue_date",
        "effective_date",
        "expiration_date",
        "permit_comp_status_flag",
        "state_water_body",
        "state_water_body_name"
    )
)

# Check permit status distribution for CAFOs
display(
    df_cafo_facilities
    .join(df_permits_clean, on="npdes_id", how="left")
    .groupBy("permit_status_code")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

df_silver = (
    df_cafo_facilities
    .join(df_permits_clean, on="npdes_id", how="left")
)

print(f"CAFO facilities with permit info: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add NAICS description for livestock type

# COMMAND ----------

naics_descriptions = {
    "112111": "Beef Cattle Ranching",
    "112112": "Cattle Feedlots",
    "112120": "Dairy Cattle and Milk Production",
    "112130": "Dual-Purpose Cattle",
    "112210": "Turkey Production",
    "112310": "Chicken Egg Production",
    "112320": "Broilers and Other Meat-Type Chicken",
    "112330": "Turkey Production",
    "112340": "Poultry Hatcheries",
    "112390": "Other Poultry Production",
    "112410": "Sheep Farming",
    "112420": "Goat Farming",
    "112511": "Finfish Farming",
    "112512": "Shellfish Farming",
    "112519": "Other Aquaculture",
    "112920": "Horses and Other Equine Production",
    "112990": "All Other Animal Production"
}

naics_desc_rows = [(k, v) for k, v in naics_descriptions.items()]
df_naics_desc = spark.createDataFrame(naics_desc_rows, ["naics_code", "livestock_type"])

df_cafo_naics_labeled = (
    df_cafo_naics
    .join(F.broadcast(df_naics_desc), on="naics_code", how="left")
    .select("npdes_id", "naics_code", "livestock_type")
    .dropDuplicates(["npdes_id"])
)

df_silver = df_silver.join(df_cafo_naics_labeled, on="npdes_id", how="left")

print(f"Final Silver row count: {df_silver.count():,}")
display(df_silver.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Silver

# COMMAND ----------

target_table = "cancer_environment_lakehouse.silver.cafo_facilities"

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_cafo_facilities,
# MAGIC     COUNT(DISTINCT state_code) AS states,
# MAGIC     SUM(CASE WHEN near_impaired_waters THEN 1 ELSE 0 END) AS near_impaired_waters,
# MAGIC     COUNT(DISTINCT naics_code) AS livestock_types
# MAGIC FROM cancer_environment_lakehouse.silver.cafo_facilities;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Top 10 states by CAFO count near impaired waters

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_code,
# MAGIC     COUNT(*) AS total_cafos,
# MAGIC     SUM(CASE WHEN near_impaired_waters THEN 1 ELSE 0 END) AS cafos_near_impaired_waters,
# MAGIC     ROUND(SUM(CASE WHEN near_impaired_waters THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_near_impaired
# MAGIC FROM cancer_environment_lakehouse.silver.cafo_facilities
# MAGIC GROUP BY state_code
# MAGIC ORDER BY cafos_near_impaired_waters DESC
# MAGIC LIMIT 10;