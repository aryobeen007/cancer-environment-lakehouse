# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Cancer Summary
# MAGIC
# MAGIC I'm building the primary Gold table that aggregates cancer incidence and mortality
# MAGIC data at the state level. This table is the anchor for all downstream joins in the
# MAGIC environmental risk profile and serves as the primary outcome variable for ML models.
# MAGIC
# MAGIC I join incidence (1999-2022) and mortality (2018-2023) on state_abbr, keeping all
# MAGIC years from both datasets. For the overlapping years (2018-2022) both metrics are
# MAGIC available; outside that window only one metric is present.
# MAGIC
# MAGIC **Source tables:** `silver.cancer_incidence`, `silver.cancer_mortality`
# MAGIC **Target table:** `gold.state_cancer_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

df_incidence = spark.table(silver_table("cancer_incidence"))
df_mortality = spark.table(silver_table("cancer_mortality"))

print(f"Incidence rows: {df_incidence.count():,}")
print(f"Mortality rows: {df_mortality.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate incidence to state-year level
# MAGIC
# MAGIC Incidence is already at state-year grain — just selecting and renaming.

# COMMAND ----------

df_inc_agg = (
    df_incidence
    .select(
        "state_name",
        "state_abbr",
        "state_fips",
        "year",
        F.col("new_cases").alias("cancer_new_cases"),
        F.col("population").alias("incidence_population"),
        F.col("age_adjusted_rate").alias("cancer_incidence_rate")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Aggregate mortality to state-year level

# COMMAND ----------

df_mort_agg = (
    df_mortality
    .select(
        "state_abbr",
        "year",
        F.col("deaths").alias("cancer_deaths"),
        F.col("population").alias("mortality_population"),
        F.col("age_adjusted_rate").alias("cancer_mortality_rate")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Full outer join on state_abbr + year
# MAGIC
# MAGIC Full outer join preserves all years from both datasets.
# MAGIC Incidence: 1999-2022, Mortality: 2018-2023.

# COMMAND ----------

df_gold = (
    df_inc_agg
    .join(df_mort_agg, on=["state_abbr", "year"], how="full")

    # Coalesce state info from either side
    .withColumn("state_name",
        F.coalesce(F.col("state_name"), F.lit(None)))
    .withColumn("state_fips",
        F.coalesce(F.col("state_fips"), F.lit(None)))

    # Compute mortality-to-incidence ratio where both are available
    .withColumn("mortality_to_incidence_ratio",
        F.when(
            F.col("cancer_incidence_rate").isNotNull() &
            F.col("cancer_mortality_rate").isNotNull() &
            (F.col("cancer_incidence_rate") > 0),
            F.round(
                F.col("cancer_mortality_rate") / F.col("cancer_incidence_rate") * 100, 2
            )
        ).otherwise(F.lit(None).cast(DoubleType())))

    # Flag data availability
    .withColumn("has_incidence_data",
        F.col("cancer_incidence_rate").isNotNull())
    .withColumn("has_mortality_data",
        F.col("cancer_mortality_rate").isNotNull())

    .orderBy("state_abbr", "year")
)

print(f"Gold rows: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

display(df_gold.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write to Gold

# COMMAND ----------

target_table = gold_table("state_cancer_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     SUM(CASE WHEN has_incidence_data AND has_mortality_data THEN 1 ELSE 0 END) AS rows_with_both,
# MAGIC     SUM(CASE WHEN has_incidence_data AND NOT has_mortality_data THEN 1 ELSE 0 END) AS incidence_only,
# MAGIC     SUM(CASE WHEN NOT has_incidence_data AND has_mortality_data THEN 1 ELSE 0 END) AS mortality_only
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Top 10 states by average cancer incidence rate (1999-2022)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(AVG(cancer_incidence_rate), 1) AS avg_incidence_rate,
# MAGIC     ROUND(AVG(cancer_mortality_rate), 1) AS avg_mortality_rate,
# MAGIC     ROUND(AVG(mortality_to_incidence_ratio), 1) AS avg_mortality_ratio_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary
# MAGIC WHERE has_incidence_data = true
# MAGIC GROUP BY state_abbr, state_name
# MAGIC ORDER BY avg_incidence_rate DESC
# MAGIC LIMIT 10;