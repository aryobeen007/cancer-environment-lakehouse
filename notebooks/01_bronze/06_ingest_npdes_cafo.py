# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze - NPDES / CAFO Facilities (15 Tables)
# MAGIC
# MAGIC I'm ingesting all 15 EPA NPDES (National Pollutant Discharge Elimination System)
# MAGIC files into the Bronze layer. These files come from the EPA ECHO (Enforcement and
# MAGIC Compliance History Online) system and cover CAFO (Concentrated Animal Feeding
# MAGIC Operation) permits, inspections, violations, and enforcement actions.
# MAGIC
# MAGIC Each file becomes its own Bronze Delta table using the same loop pattern I used
# MAGIC for the SDWIS water quality ingestion.
# MAGIC
# MAGIC **Source files:** 15 ICIS_*.csv and NPDES_*.csv files
# MAGIC **Target tables:** `cancer_environment_lakehouse.bronze.icis_*` and `bronze.npdes_*`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup - Define the 15 NPDES files to ingest

# COMMAND ----------

import re
from pyspark.sql import functions as F

def sanitize_column_name(col_name: str) -> str:
    name = col_name.strip().lower()
    name = re.sub(r"[ \-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

# Ordered smallest to largest based on file sizes from the source folder
NPDES_DATASETS = {
    "npdes_formal_enforcement_actions":   "NPDES_FORMAL_ENFORCEMENT_ACTIONS.csv",
    "npdes_cs_violations":                "NPDES_CS_VIOLATIONS.csv",
    "npdes_se_violations":                "NPDES_SE_VIOLATIONS.csv",
    "npdes_perm_feature_coords":          "NPDES_PERM_FEATURE_COORDS.csv",
    "npdes_perm_components":              "NPDES_PERM_COMPONENTS.csv",
    "npdes_naics":                        "NPDES_NAICS.csv",
    "npdes_sics":                         "NPDES_SICS.csv",
    "npdes_ps_violations":                "NPDES_PS_VIOLATIONS.csv",
    "npdes_informal_enforcement_actions": "NPDES_INFORMAL_ENFORCEMENT_ACTIONS.csv",
    "npdes_data_groups":                  "NPDES_DATA_GROUPS.csv",
    "npdes_inspections":                  "NPDES_INSPECTIONS.csv",
    "icis_facilities":                    "ICIS_FACILITIES.csv",
    "npdes_violation_enforcements":       "NPDES_VIOLATION_ENFORCEMENTS.csv",
    "npdes_qncr_history":                 "NPDES_QNCR_HISTORY.csv",
    "icis_permits":                       "ICIS_PERMITS.csv",
}

print(f"NPDES datasets to ingest: {len(NPDES_DATASETS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest all 15 NPDES files

# COMMAND ----------

ingestion_summary = []

for table_key, filename in NPDES_DATASETS.items():
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
print("NPDES/CAFO BRONZE INGESTION SUMMARY")
print("="*70)
total_rows = 0
for item in ingestion_summary:
    status_icon = "✅" if item["status"] == "SUCCESS" else "❌"
    print(f"{status_icon} {item['file']:<50} {item['rows']:>12,} rows  {item['status']}")
    total_rows += item["rows"]
print("-"*70)
print(f"Total rows ingested: {total_rows:,}")
print(f"Tables succeeded: {sum(1 for i in ingestion_summary if i['status'] == 'SUCCESS')}/{len(ingestion_summary)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate all 15 Bronze tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'icis_facilities' AS table_name, COUNT(*) AS row_count FROM cancer_environment_lakehouse.bronze.icis_facilities
# MAGIC UNION ALL
# MAGIC SELECT 'icis_permits', COUNT(*) FROM cancer_environment_lakehouse.bronze.icis_permits
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_cs_violations', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_cs_violations
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_data_groups', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_data_groups
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_formal_enforcement_actions', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_formal_enforcement_actions
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_informal_enforcement_actions', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_informal_enforcement_actions
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_inspections', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_inspections
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_naics', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_naics
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_perm_components', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_perm_components
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_perm_feature_coords', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_perm_feature_coords
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_ps_violations', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_ps_violations
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_qncr_history', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_qncr_history
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_se_violations', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_se_violations
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_sics', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_sics
# MAGIC UNION ALL
# MAGIC SELECT 'npdes_violation_enforcements', COUNT(*) FROM cancer_environment_lakehouse.bronze.npdes_violation_enforcements
# MAGIC ORDER BY table_name;