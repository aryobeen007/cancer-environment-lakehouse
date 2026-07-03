# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - SDWIS Water Quality
# MAGIC
# MAGIC I'm cleaning and transforming two key SDWIS Bronze tables into Silver:
# MAGIC
# MAGIC 1. **`sdwa_violations_enforcement`** (15.3M rows) — I filter to health-based violations
# MAGIC    only (`is_health_based_ind = 'Y'`) since those are the ones with direct public health
# MAGIC    impact and most relevant to cancer-environment analysis.
# MAGIC
# MAGIC 2. **`sdwa_pub_water_systems`** (433K rows) — I clean and standardize the water system
# MAGIC    inventory, keeping active systems only and extracting state codes for joining.
# MAGIC
# MAGIC The join key between these two tables is `pwsid` (Public Water System ID).
# MAGIC The join key to cancer tables is `state_code` / `primacy_agency_code`.
# MAGIC
# MAGIC **Source tables:** `bronze.sdwa_violations_enforcement`, `bronze.sdwa_pub_water_systems`
# MAGIC **Target tables:** `silver.sdwa_violations`, `silver.sdwa_water_systems`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 — Water Systems (smaller table first)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Read and inspect Bronze water systems

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DateType

df_systems_bronze = spark.table(bronze_table("sdwa_pub_water_systems"))
print(f"Bronze water systems row count: {df_systems_bronze.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Filter to active systems and select key columns

# COMMAND ----------

# Check active vs inactive systems
display(
    df_systems_bronze.groupBy("pws_activity_code")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

df_systems_silver = (
    df_systems_bronze
    # Keep only active water systems
    .filter(F.col("pws_activity_code") == "A")

    # Select analytically relevant columns
    .select(
        "pwsid",
        "pws_name",
        F.col("primacy_agency_code").alias("state_code"),
        "epa_region",
        "pws_type_code",
        "primary_source_code",
        "population_served_count",
        "service_connections_count",
        "owner_type_code",
        "is_school_or_daycare_ind",
        "gw_sw_code",
        "city_name",
        "zip_code"
    )

    # Cast types
    .withColumn("population_served_count",
        F.col("population_served_count").cast(IntegerType()))
    .withColumn("service_connections_count",
        F.col("service_connections_count").cast(IntegerType()))

    # Add water source description
    .withColumn("water_source",
        F.when(F.col("gw_sw_code") == "GW", "Ground Water")
        .when(F.col("gw_sw_code") == "SW", "Surface Water")
        .when(F.col("gw_sw_code") == "GU", "Ground Water Under Influence")
        .otherwise(F.col("gw_sw_code")))

    # Add system type description
    .withColumn("system_type",
        F.when(F.col("pws_type_code") == "CWS", "Community Water System")
        .when(F.col("pws_type_code") == "NTNCWS", "Non-Transient Non-Community")
        .when(F.col("pws_type_code") == "TNCWS", "Transient Non-Community")
        .otherwise(F.col("pws_type_code")))
)

print(f"Active water systems: {df_systems_silver.count():,}")
display(df_systems_silver.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c. Write water systems to Silver

# COMMAND ----------

target_systems = silver_table("sdwa_water_systems")
df_systems_silver.write.format("delta").mode("overwrite").saveAsTable(target_systems)
print(f"Written to {target_systems}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2 — Violations (15.3M rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Read Bronze violations and check health-based distribution

# COMMAND ----------

df_viol_bronze = spark.table(bronze_table("sdwa_violations_enforcement"))
print(f"Bronze violations row count: {df_viol_bronze.count():,}")

# Check health-based vs non-health-based split
display(
    df_viol_bronze.groupBy("is_health_based_ind")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Filter to health-based violations only
# MAGIC
# MAGIC Health-based violations (is_health_based_ind = 'Y') are those where a contaminant
# MAGIC exceeded the Maximum Contaminant Level (MCL) or a treatment technique was violated.
# MAGIC These have direct public health implications and are what I want to correlate with
# MAGIC cancer rates. Monitoring and reporting violations are excluded.

# COMMAND ----------

df_viol_silver = (
    df_viol_bronze
    .filter(F.col("is_health_based_ind") == "Y")

    # Select key columns
    .select(
        "pwsid",
        "violation_id",
        "violation_code",
        "violation_category_code",
        "is_health_based_ind",
        "contaminant_code",
        "viol_measure",
        "unit_of_measure",
        "federal_mcl",
        "is_major_viol_ind",
        "severity_ind_cnt",
        "violation_status",
        "public_notification_tier",
        "rule_code",
        "rule_group_code",
        "rule_family_code",
        "compl_per_begin_date",
        "compl_per_end_date",
        "non_compl_per_begin_date",
        "viol_first_reported_date",
        "viol_last_reported_date",
        "enforcement_id",
        "enforcement_date",
        "enforcement_action_type_code",
        "enf_action_category"
    )

    # Extract year from violation first reported date for time-series joins
    .withColumn("violation_year",
        F.year(F.col("viol_first_reported_date")))

    # Flag major violations
    .withColumn("is_major_violation",
        F.when(F.col("is_major_viol_ind") == "Y", True).otherwise(False))

    # Flag if enforcement action was taken
    .withColumn("has_enforcement",
        F.when(F.col("enforcement_id").isNotNull(), True).otherwise(False))

    .drop("is_health_based_ind", "is_major_viol_ind")
)

print(f"Health-based violations: {df_viol_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c. Check violation category distribution

# COMMAND ----------

display(
    df_viol_silver.groupBy("violation_category_code")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d. Write violations to Silver

# COMMAND ----------

target_viol = silver_table("sdwa_violations")
df_viol_silver.write.format("delta").mode("overwrite").saveAsTable(target_viol)
print(f"Written to {target_viol}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3 — Validate both Silver tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'sdwa_water_systems' AS table_name, COUNT(*) AS row_count
# MAGIC FROM cancer_environment_lakehouse.silver.sdwa_water_systems
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_violations', COUNT(*)
# MAGIC FROM cancer_environment_lakehouse.silver.sdwa_violations;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4 — Preview: top 10 states by health-based violation count

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ws.state_code,
# MAGIC     COUNT(v.violation_id) AS health_based_violations,
# MAGIC     COUNT(DISTINCT v.pwsid) AS affected_systems,
# MAGIC     COUNT(DISTINCT v.contaminant_code) AS distinct_contaminants,
# MAGIC     SUM(CASE WHEN v.has_enforcement THEN 1 ELSE 0 END) AS enforced_violations
# MAGIC FROM cancer_environment_lakehouse.silver.sdwa_violations v
# MAGIC JOIN cancer_environment_lakehouse.silver.sdwa_water_systems ws
# MAGIC     ON v.pwsid = ws.pwsid
# MAGIC GROUP BY ws.state_code
# MAGIC ORDER BY health_based_violations DESC
# MAGIC LIMIT 10;