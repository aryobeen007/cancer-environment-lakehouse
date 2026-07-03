# Databricks notebook source
# MAGIC %md
# MAGIC # Silver - CDC Chronic Disease Indicators
# MAGIC
# MAGIC I'm cleaning and transforming the Bronze chronic disease table into a production-ready
# MAGIC Silver table. This is the most complex Silver transformation so far because the dataset
# MAGIC is long-format (one row per state × year × topic × question × stratification) and
# MAGIC needs careful filtering to extract the state-level overall indicators I need.
# MAGIC
# MAGIC Key decisions I'm making here:
# MAGIC
# MAGIC 1. **Filter to Overall stratification only** — the dataset has breakdowns by Sex, Age,
# MAGIC    Race/Ethnicity. I keep only "Overall" rows for state-level analysis. Demographic
# MAGIC    breakdowns can be explored separately later.
# MAGIC 2. **Filter to relevant topics** — I keep the 7 topics most relevant to cancer-environment
# MAGIC    analysis: Cancer, Tobacco, Cardiovascular Disease, Diabetes, COPD, Nutrition/Physical
# MAGIC    Activity/Weight, and Asthma.
# MAGIC 3. **Keep Age-Adjusted Prevalence where available** — for comparability across states
# MAGIC    with different age distributions. Fall back to Crude Prevalence where age-adjusted
# MAGIC    isn't available.
# MAGIC 4. **Handle suppressed values** — rows with DataValueFootnoteSymbol = '*' have no
# MAGIC    data value. I flag these as suppressed rather than dropping them.
# MAGIC 5. **Standardize column names** — align to the same conventions as other Silver tables.
# MAGIC
# MAGIC **Source table:** `cancer_environment_lakehouse.bronze.chronic_disease`
# MAGIC **Target table:** `cancer_environment_lakehouse.silver.chronic_disease`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read from Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

df_bronze = spark.table(bronze_table("chronic_disease"))

print(f"Bronze row count: {df_bronze.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filter to relevant topics

# COMMAND ----------

RELEVANT_TOPICS = [
    "Cancer",
    "Tobacco",
    "Cardiovascular Disease",
    "Diabetes",
    "Chronic Obstructive Pulmonary Disease",
    "Nutrition, Physical Activity, and Weight Status",
    "Asthma"
]

df_filtered = df_bronze.filter(F.col("topic").isin(RELEVANT_TOPICS))

print(f"Rows after topic filter: {df_filtered.count():,}")

display(
    df_filtered.groupBy("topic")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter to Overall stratification only
# MAGIC
# MAGIC I keep rows where stratification_category1 = 'Overall' and stratification1 = 'Overall'.
# MAGIC This gives one row per state × year × question combination.

# COMMAND ----------

df_overall = df_filtered.filter(
    (F.col("stratification_category1") == "Overall") &
    (F.col("stratification1") == "Overall")
)

print(f"Rows after Overall stratification filter: {df_overall.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Keep Age-Adjusted Prevalence where available
# MAGIC
# MAGIC For each state × year × question, I prefer Age-Adjusted Prevalence over Crude
# MAGIC Prevalence for cross-state comparability. I use a window function to rank by
# MAGIC data_value_type and keep the best available.

# COMMAND ----------

from pyspark.sql.window import Window

# Rank data value types — Age-adjusted is preferred (rank 1), Crude is fallback (rank 2)
# Other types (Number, Mean) are kept as-is since they don't have age-adjusted alternatives
preferred_types = {
    "Age-adjusted Prevalence": 1,
    "Age-adjusted Mean": 1,
    "Age-adjusted Rate": 1,
    "Crude Prevalence": 2,
    "Crude Mean": 2,
    "Crude Rate": 2,
}

# Create a preference column
preference_expr = F.coalesce(
    *[F.when(F.col("data_value_type") == k, F.lit(v)) for k, v in preferred_types.items()],
    F.lit(3)  # Everything else gets rank 3
)

df_ranked = df_overall.withColumn("type_rank", preference_expr)

# For each state × year × question, keep only the highest-ranked (lowest number) data type
window = Window.partitionBy(
    "location_abbr", "year_start", "question_id"
).orderBy("type_rank")

df_deduped = (
    df_ranked
    .withColumn("row_num", F.row_number().over(window))
    .filter(F.col("row_num") == 1)
    .drop("row_num", "type_rank")
)

print(f"Rows after deduplication: {df_deduped.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Flag suppressed values and cast types

# COMMAND ----------

df_silver = (
    df_deduped
    # Flag suppressed values
    .withColumn("is_suppressed",
        F.when(F.col("data_value_footnote_symbol").isNotNull(), True)
        .otherwise(False))

    # Cast types
    .withColumn("year_start", F.col("year_start").cast(IntegerType()))
    .withColumn("year_end", F.col("year_end").cast(IntegerType()))
    .withColumn("data_value", F.col("data_value").cast(DoubleType()))
    .withColumn("low_confidence_limit", F.col("low_confidence_limit").cast(DoubleType()))
    .withColumn("high_confidence_limit", F.col("high_confidence_limit").cast(DoubleType()))

    # Rename for clarity
    .withColumnRenamed("location_abbr", "state_abbr")
    .withColumnRenamed("location_desc", "state_name")
    .withColumnRenamed("year_start", "year")

    # Drop columns not needed in Silver
    .drop(
        "year_end", "response", "data_value_alt", "data_value_footnote_symbol",
        "data_value_footnote", "stratification_category1", "stratification1",
        "stratification_category2", "stratification2", "stratification_category3",
        "stratification3", "stratification_category_id1", "stratification_id1",
        "stratification_category_id2", "stratification_id2",
        "stratification_category_id3", "stratification_id3",
        "location_id", "response_id", "ingested_at", "source_file"
    )
)

print(f"Final column count: {len(df_silver.columns)}")
df_silver.printSchema()

# COMMAND ----------

display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Filter out non-state rows
# MAGIC
# MAGIC The dataset includes US-level and territory rows (PR, GU, VI).
# MAGIC I keep only the 51 US states + DC.

# COMMAND ----------

valid_states = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
]

df_silver = df_silver.filter(F.col("state_abbr").isin(valid_states))

print(f"Rows after state filter: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write to Silver

# COMMAND ----------

target_table = silver_table("chronic_disease")

df_silver.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data quality summary

# COMMAND ----------

data_quality_summary(df_silver, target_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Validate — coverage and topic breakdown

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     COUNT(DISTINCT topic) AS topics,
# MAGIC     COUNT(DISTINCT question_id) AS questions,
# MAGIC     SUM(CASE WHEN is_suppressed THEN 1 ELSE 0 END) AS suppressed_rows
# MAGIC FROM cancer_environment_lakehouse.silver.chronic_disease;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Preview — smoking prevalence by state (most recent year)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     year,
# MAGIC     ROUND(data_value, 1) AS smoking_prevalence_pct,
# MAGIC     data_value_type
# MAGIC FROM cancer_environment_lakehouse.silver.chronic_disease
# MAGIC WHERE topic = 'Tobacco'
# MAGIC   AND question_id = 'TOB04'
# MAGIC   AND year = (SELECT MAX(year) FROM cancer_environment_lakehouse.silver.chronic_disease
# MAGIC               WHERE topic = 'Tobacco' AND question_id = 'TOB04')
# MAGIC   AND is_suppressed = false
# MAGIC ORDER BY smoking_prevalence_pct DESC
# MAGIC LIMIT 15;