# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - SDWIS Water Quality (11 Tables)
# MAGIC
# MAGIC I'm ingesting all 11 EPA Safe Drinking Water Information System (SDWIS) files into
# MAGIC the Bronze layer. Each file becomes its own Bronze Delta table. I use a loop to
# MAGIC apply the same read → sanitize → add metadata → write pattern to every file,
# MAGIC keeping the code clean and consistent.
# MAGIC
# MAGIC The largest file is SDWA_VIOLATIONS_ENFORCEMENT.csv (~3.97 GB, 15M+ rows) which
# MAGIC will take a few minutes to process. All others are under 400 MB.
# MAGIC
# MAGIC **Source files:** 11 SDWA_*.csv files
# MAGIC **Target tables:** `cancer_environment_lakehouse.bronze.sdwa_*`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup - Define the 11 SDWIS files to ingest

# COMMAND ----------

import re
from pyspark.sql import functions as F

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

# Map each SDWIS table key to its source filename and Bronze table name
SDWIS_DATASETS = {
    "sdwa_events_milestones":       "SDWA_EVENTS_MILESTONES.csv",
    "sdwa_ref_ansi_areas":          "SDWA_REF_ANSI_AREAS.csv",
    "sdwa_ref_code_values":         "SDWA_REF_CODE_VALUES.csv",
    "sdwa_service_areas":           "SDWA_SERVICE_AREAS.csv",
    "sdwa_geographic_areas":        "SDWA_GEOGRAPHIC_AREAS.csv",
    "sdwa_pn_violation_assoc":      "SDWA_PN_VIOLATION_ASSOC.csv",
    "sdwa_facilities":              "SDWA_FACILITIES.csv",
    "sdwa_lcr_samples":             "SDWA_LCR_SAMPLES.csv",
    "sdwa_pub_water_systems":       "SDWA_PUB_WATER_SYSTEMS.csv",
    "sdwa_site_visits":             "SDWA_SITE_VISITS.csv",
    "sdwa_violations_enforcement":  "SDWA_VIOLATIONS_ENFORCEMENT.csv",  # largest - runs last
}

print(f"SDWIS datasets to ingest: {len(SDWIS_DATASETS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest all 11 SDWIS files
# MAGIC
# MAGIC I process them smallest to largest so any issues with the big files don't block
# MAGIC the smaller ones. SDWA_VIOLATIONS_ENFORCEMENT.csv runs last since it's ~3.97 GB.

# COMMAND ----------

ingestion_summary = []

for table_key, filename in SDWIS_DATASETS.items():
    source_path = f"{LANDING_VOLUME_PATH}/{filename}"
    target_table = BRONZE_TABLES[table_key]

    print(f"\n{'='*60}")
    print(f"Ingesting: {filename}")
    print(f"Target:    {target_table}")
    print(f"{'='*60}")

    try:
        # Read
        df_raw = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(source_path)
        )

        row_count = df_raw.count()
        print(f"Read {row_count:,} rows")

        # Sanitize column names
        renamed = {col: sanitize_column_name(col) for col in df_raw.columns}
        for old, new in renamed.items():
            df_raw = df_raw.withColumnRenamed(old, new)

        # Add metadata
        df_bronze = add_ingestion_metadata(df_raw, filename)

        # Write
        write_bronze_table(df_bronze, target_table)

        ingestion_summary.append({
            "table": target_table,
            "file": filename,
            "rows": row_count,
            "status": "SUCCESS"
        })

    except Exception as e:
        print(f"ERROR ingesting {filename}: {e}")
        ingestion_summary.append({
            "table": target_table,
            "file": filename,
            "rows": 0,
            "status": f"FAILED: {str(e)}"
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Summary

# COMMAND ----------

print("\n" + "="*70)
print("SDWIS BRONZE INGESTION SUMMARY")
print("="*70)
total_rows = 0
for item in ingestion_summary:
    status_icon = "✅" if item["status"] == "SUCCESS" else "❌"
    print(f"{status_icon} {item['file']:<45} {item['rows']:>12,} rows  {item['status']}")
    total_rows += item["rows"]
print("-"*70)
print(f"Total rows ingested: {total_rows:,}")
print(f"Tables succeeded: {sum(1 for i in ingestion_summary if i['status'] == 'SUCCESS')}/{len(ingestion_summary)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate all 11 Bronze tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'sdwa_events_milestones' AS table_name, COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.sdwa_events_milestones
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_ref_ansi_areas', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_ref_ansi_areas
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_ref_code_values', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_ref_code_values
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_service_areas', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_service_areas
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_geographic_areas', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_geographic_areas
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_pn_violation_assoc', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_pn_violation_assoc
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_facilities', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_facilities
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_lcr_samples', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_lcr_samples
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_pub_water_systems', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_pub_water_systems
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_site_visits', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_site_visits
# MAGIC UNION ALL
# MAGIC SELECT 'sdwa_violations_enforcement', COUNT(*) FROM cancer_environment_lakehouse.bronze.sdwa_violations_enforcement
# MAGIC ORDER BY table_name;