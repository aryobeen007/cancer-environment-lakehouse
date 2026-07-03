# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State CAFO Summary
# MAGIC
# MAGIC I'm aggregating CAFO facility data to state level for the Gold layer.
# MAGIC Key metrics include total CAFO count, facilities near impaired waters,
# MAGIC livestock type breakdown, and permit compliance status.
# MAGIC
# MAGIC **Source table:** `silver.cafo_facilities`
# MAGIC **Target table:** `gold.state_cafo_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver table

# COMMAND ----------

from pyspark.sql import functions as F

df_cafo = spark.table("cancer_environment_lakehouse.silver.cafo_facilities")

print(f"CAFO facilities: {df_cafo.count():,}")
display(df_cafo.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate to state level

# COMMAND ----------

df_gold = (
    df_cafo
    .filter(F.col("state_code").isNotNull())
    .groupBy("state_code")
    .agg(
        # Total facilities
        F.count("npdes_id").alias("total_cafo_facilities"),
        F.countDistinct("npdes_id").alias("distinct_cafo_permits"),

        # Impaired waters
        F.sum(F.when(F.col("near_impaired_waters") == True, 1).otherwise(0))
            .alias("cafos_near_impaired_waters"),
        F.round(
            F.sum(F.when(F.col("near_impaired_waters") == True, 1).otherwise(0)) * 100.0 /
            F.count("npdes_id"), 1
        ).alias("pct_near_impaired_waters"),

        # Permit status
        F.sum(F.when(F.col("permit_status_code") == "EFF", 1).otherwise(0))
            .alias("active_permits"),
        F.sum(F.when(F.col("permit_status_code") == "EXP", 1).otherwise(0))
            .alias("expired_permits"),

        # Livestock type breakdown
        F.sum(F.when(F.col("naics_code").isin(
            "112111", "112112", "112130"), 1).otherwise(0)).alias("cattle_facilities"),
        F.sum(F.when(F.col("naics_code") == "112120", 1).otherwise(0))
            .alias("dairy_facilities"),
        F.sum(F.when(F.col("naics_code").isin(
            "112310", "112320", "112330", "112340", "112390"), 1).otherwise(0))
            .alias("poultry_facilities"),
        F.sum(F.when(F.col("naics_code").isin(
            "112210"), 1).otherwise(0)).alias("turkey_facilities"),
        F.sum(F.when(F.col("naics_code").isin(
            "112511", "112512", "112519"), 1).otherwise(0)).alias("aquaculture_facilities"),

        # Geographic info
        F.countDistinct("county_code").alias("counties_with_cafos"),

        # Coordinates available (for mapping)
        F.sum(F.when(
            F.col("geocode_latitude").isNotNull() &
            F.col("geocode_longitude").isNotNull(), 1).otherwise(0))
            .alias("facilities_with_coordinates")
    )
    .withColumnRenamed("state_code", "state_abbr")
    .orderBy("total_cafo_facilities", ascending=False)
)

print(f"Gold rows: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

display(df_gold.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write to Gold

# COMMAND ----------

target_table = gold_table("state_cafo_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     SUM(total_cafo_facilities) AS total_cafos,
# MAGIC     SUM(cafos_near_impaired_waters) AS total_near_impaired,
# MAGIC     ROUND(AVG(pct_near_impaired_waters), 1) AS avg_pct_near_impaired
# MAGIC FROM cancer_environment_lakehouse.gold.state_cafo_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Top 10 states by CAFOs near impaired waters

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     total_cafo_facilities,
# MAGIC     cafos_near_impaired_waters,
# MAGIC     pct_near_impaired_waters,
# MAGIC     cattle_facilities,
# MAGIC     dairy_facilities,
# MAGIC     poultry_facilities
# MAGIC FROM cancer_environment_lakehouse.gold.state_cafo_summary
# MAGIC ORDER BY cafos_near_impaired_waters DESC
# MAGIC LIMIT 10;