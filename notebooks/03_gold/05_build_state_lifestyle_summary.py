# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Lifestyle Summary
# MAGIC
# MAGIC I'm pivoting key lifestyle and behavioral risk indicators from the CDC Chronic
# MAGIC Disease Silver table into a wide-format state-year summary for the Gold layer.
# MAGIC
# MAGIC I extract the most analytically relevant indicators for cancer-environment analysis:
# MAGIC - Current smoking prevalence (TOB04)
# MAGIC - Obesity prevalence (NWS09 or similar)
# MAGIC - Diabetes prevalence (DIA01)
# MAGIC - Physical inactivity (PA01)
# MAGIC - COPD prevalence (COPD01)
# MAGIC - Asthma prevalence (AST02)
# MAGIC
# MAGIC **Source table:** `silver.chronic_disease`
# MAGIC **Target table:** `gold.state_lifestyle_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver table and check available question IDs

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df_cd = spark.table(silver_table("chronic_disease"))

print(f"Silver chronic disease rows: {df_cd.count():,}")

# COMMAND ----------

# Check available question IDs per topic for our key indicators
display(
    df_cd.select("topic", "question_id", "question", "data_value_type")
    .distinct()
    .orderBy("topic", "question_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define key indicators to pivot

# COMMAND ----------

KEY_INDICATORS = {
    "TOB04": "smoking_prevalence_pct",
    "DIA01": "diabetes_prevalence_pct",
    "AST02": "asthma_prevalence_pct",
    "COPD01": "copd_prevalence_pct",
}

print("Key indicators to pivot:")
for qid, name in KEY_INDICATORS.items():
    print(f"  {qid} -> {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter to key indicators and pivot to wide format

# COMMAND ----------

df_filtered = df_cd.filter(
    F.col("question_id").isin(list(KEY_INDICATORS.keys())) &
    (F.col("is_suppressed") == False)
)

print(f"Filtered rows: {df_filtered.count():,}")

# COMMAND ----------

# Pivot — one column per indicator
df_pivoted = (
    df_filtered
    .groupBy("state_abbr", "state_name", "year")
    .pivot("question_id", list(KEY_INDICATORS.keys()))
    .agg(F.first("data_value"))
)

# Rename pivot columns to readable names
for qid, col_name in KEY_INDICATORS.items():
    if qid in df_pivoted.columns:
        df_pivoted = df_pivoted.withColumnRenamed(qid, col_name)
    else:
        df_pivoted = df_pivoted.withColumn(col_name, F.lit(None).cast(DoubleType()))

print(f"Pivoted rows: {df_pivoted.count():,}")
df_pivoted.printSchema()

# COMMAND ----------

display(df_pivoted.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Add nutrition/obesity and physical inactivity if available
# MAGIC
# MAGIC Check what NWS and PA question IDs exist in the data.

# COMMAND ----------

display(
    df_cd.filter(F.col("topic").isin(
        ["Nutrition, Physical Activity, and Weight Status"]))
    .select("question_id", "question")
    .distinct()
    .orderBy("question_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add obesity and physical inactivity metrics

# COMMAND ----------

EXTRA_INDICATORS = {
    "NPW14": "obesity_prevalence_pct",
    "NPW06": "physical_inactivity_pct",
}

df_extra = df_cd.filter(
    F.col("question_id").isin(list(EXTRA_INDICATORS.keys())) &
    (F.col("is_suppressed") == False)
)

if df_extra.count() > 0:
    df_extra_pivoted = (
        df_extra
        .groupBy("state_abbr", "year")
        .pivot("question_id", list(EXTRA_INDICATORS.keys()))
        .agg(F.first("data_value"))
    )
    for qid, col_name in EXTRA_INDICATORS.items():
        if qid in df_extra_pivoted.columns:
            df_extra_pivoted = df_extra_pivoted.withColumnRenamed(qid, col_name)
        else:
            df_extra_pivoted = df_extra_pivoted.withColumn(
                col_name, F.lit(None).cast(DoubleType()))

    df_gold = df_pivoted.join(df_extra_pivoted, on=["state_abbr", "year"], how="left")
    print(f"Added extra indicators. Final rows: {df_gold.count():,}")
else:
    df_gold = df_pivoted
    for col_name in EXTRA_INDICATORS.values():
        df_gold = df_gold.withColumn(col_name, F.lit(None).cast(DoubleType()))
    print("Extra indicators not found — added as null columns")

# COMMAND ----------

df_gold = df_gold.orderBy("state_abbr", "year")
print(f"Final Gold rows: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Gold

# COMMAND ----------

target_table = gold_table("state_lifestyle_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     ROUND(AVG(smoking_prevalence_pct), 1) AS national_avg_smoking,
# MAGIC     ROUND(AVG(diabetes_prevalence_pct), 1) AS national_avg_diabetes,
# MAGIC     ROUND(AVG(obesity_prevalence_pct), 1) AS national_avg_obesity
# MAGIC FROM cancer_environment_lakehouse.gold.state_lifestyle_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Most recent year — state lifestyle profile

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     year,
# MAGIC     ROUND(smoking_prevalence_pct, 1) AS smoking_pct,
# MAGIC     ROUND(obesity_prevalence_pct, 1) AS obesity_pct,
# MAGIC     ROUND(diabetes_prevalence_pct, 1) AS diabetes_pct,
# MAGIC     ROUND(asthma_prevalence_pct, 1) AS asthma_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_lifestyle_summary
# MAGIC WHERE year = (SELECT MAX(year) FROM cancer_environment_lakehouse.gold.state_lifestyle_summary)
# MAGIC ORDER BY smoking_pct DESC
# MAGIC LIMIT 15;