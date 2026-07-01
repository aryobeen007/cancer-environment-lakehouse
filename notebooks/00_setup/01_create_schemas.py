# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Create Catalog, Schemas & Landing Volume
# MAGIC
# MAGIC I'm setting up the Unity Catalog structure for the Cancer & Environment Lakehouse project.
# MAGIC This mirrors the medallion pattern I used in my `medicare_provider_quality` project: a
# MAGIC dedicated catalog with `raw`, `bronze`, `silver`, and `gold` schemas, plus a managed volume
# MAGIC for landing my raw CSVs before ingestion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the catalog

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS cancer_environment_lakehouse
# MAGIC COMMENT 'Cancer & Environment Lakehouse - exploring associations between U.S. cancer rates and environmental, lifestyle, and water quality factors.';

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG cancer_environment_lakehouse;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the schemas
# MAGIC
# MAGIC - `raw` - holds the landing volume where I upload source CSVs
# MAGIC - `bronze` - raw ingested Delta tables, untouched except for metadata columns
# MAGIC - `silver` - cleaned, typed, validated Delta tables
# MAGIC - `gold` - aggregated, business-ready Delta tables for analytics/ML/Tableau

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS cancer_environment_lakehouse.raw
# MAGIC COMMENT 'Landing zone for raw source CSV files before ingestion.';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS cancer_environment_lakehouse.bronze
# MAGIC COMMENT 'Raw ingested Delta tables - one per source dataset, minimal transformation.';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS cancer_environment_lakehouse.silver
# MAGIC COMMENT 'Cleaned, typed, and validated Delta tables.';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS cancer_environment_lakehouse.gold
# MAGIC COMMENT 'Aggregated, business-ready Delta tables for analytics, ML, and Tableau.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the landing volume
# MAGIC
# MAGIC I'll upload my 9 source CSVs into this volume through the Catalog UI (drag-and-drop),
# MAGIC the same way I did for the `landing` volume in my Medicare project.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS cancer_environment_lakehouse.raw.landing
# MAGIC COMMENT 'Landing zone for the 9 raw source CSV files (cancer, AQI, water, food, CAFO, etc).';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify the structure

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW SCHEMAS IN cancer_environment_lakehouse;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VOLUMES IN cancer_environment_lakehouse.raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Confirm the volume path
# MAGIC
# MAGIC This is the path I'll reference in `00_workspace_config.py` and in every Bronze
# MAGIC ingestion notebook going forward.

# COMMAND ----------

landing_volume_path = "/Volumes/cancer_environment_lakehouse/raw/landing"
print(f"Landing volume path: {landing_volume_path}")

# Confirm it's accessible
dbutils.fs.ls(landing_volume_path)