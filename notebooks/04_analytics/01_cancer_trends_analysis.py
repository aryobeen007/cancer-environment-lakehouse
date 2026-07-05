# Databricks notebook source
# MAGIC %md
# MAGIC # Analytics 01 - Cancer Trends Analysis (1999-2022)
# MAGIC
# MAGIC I'm analyzing cancer incidence and mortality trends across U.S. states from
# MAGIC 1999 to 2022. This notebook establishes the baseline patterns I'll use to
# MAGIC contextualize the environmental factor correlations in subsequent notebooks.
# MAGIC
# MAGIC Key questions I'm answering here:
# MAGIC - How have national cancer incidence and mortality rates trended over 24 years?
# MAGIC - Which states show the highest and lowest rates consistently?
# MAGIC - Are there states where incidence is high but mortality is low (or vice versa)?
# MAGIC - How does the mortality-to-incidence ratio vary across states?
# MAGIC
# MAGIC **Source table:** `gold.state_cancer_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. National cancer incidence trend (1999-2022)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     year,
# MAGIC     ROUND(AVG(cancer_incidence_rate), 1) AS national_avg_incidence_rate,
# MAGIC     ROUND(MIN(cancer_incidence_rate), 1) AS min_state_rate,
# MAGIC     ROUND(MAX(cancer_incidence_rate), 1) AS max_state_rate,
# MAGIC     ROUND(MAX(cancer_incidence_rate) - MIN(cancer_incidence_rate), 1) AS rate_spread,
# MAGIC     COUNT(DISTINCT state_abbr) AS states_reporting
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary
# MAGIC WHERE has_incidence_data = true
# MAGIC GROUP BY year
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. National cancer mortality trend (2018-2023)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     year,
# MAGIC     ROUND(AVG(cancer_mortality_rate), 1) AS national_avg_mortality_rate,
# MAGIC     ROUND(MIN(cancer_mortality_rate), 1) AS min_state_rate,
# MAGIC     ROUND(MAX(cancer_mortality_rate), 1) AS max_state_rate,
# MAGIC     COUNT(DISTINCT state_abbr) AS states_reporting
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary
# MAGIC WHERE has_mortality_data = true
# MAGIC GROUP BY year
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Long-term state rankings — average incidence rate (1999-2022)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(AVG(cancer_incidence_rate), 1) AS avg_incidence_rate,
# MAGIC     ROUND(MIN(cancer_incidence_rate), 1) AS min_rate,
# MAGIC     ROUND(MAX(cancer_incidence_rate), 1) AS max_rate,
# MAGIC     ROUND(MAX(cancer_incidence_rate) - MIN(cancer_incidence_rate), 1) AS rate_range,
# MAGIC     COUNT(year) AS years_of_data
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary
# MAGIC WHERE has_incidence_data = true
# MAGIC GROUP BY state_abbr, state_name
# MAGIC ORDER BY avg_incidence_rate DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Mortality-to-incidence ratio by state

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     state_name,
# MAGIC     ROUND(AVG(cancer_incidence_rate), 1) AS avg_incidence_rate,
# MAGIC     ROUND(AVG(cancer_mortality_rate), 1) AS avg_mortality_rate,
# MAGIC     ROUND(AVG(mortality_to_incidence_ratio), 1) AS avg_mortality_ratio_pct,
# MAGIC     CASE
# MAGIC         WHEN AVG(mortality_to_incidence_ratio) >= 33 THEN 'High Mortality Burden'
# MAGIC         WHEN AVG(mortality_to_incidence_ratio) >= 28 THEN 'Moderate Mortality Burden'
# MAGIC         ELSE 'Lower Mortality Burden'
# MAGIC     END AS mortality_burden_category
# MAGIC FROM cancer_environment_lakehouse.gold.state_cancer_summary
# MAGIC WHERE has_incidence_data = true AND has_mortality_data = true
# MAGIC GROUP BY state_abbr, state_name
# MAGIC ORDER BY avg_mortality_ratio_pct DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. States with declining vs increasing incidence trends
# MAGIC
# MAGIC I'm comparing early period (1999-2005) vs late period (2016-2022) averages
# MAGIC to identify states where cancer rates are improving or worsening over time.

# COMMAND ----------

from pyspark.sql import functions as F

df_cancer = spark.table(gold_table("state_cancer_summary"))

df_early = (
    df_cancer
    .filter(F.col("year").between(1999, 2005))
    .filter(F.col("has_incidence_data") == True)
    .groupBy("state_abbr", "state_name")
    .agg(F.round(F.avg("cancer_incidence_rate"), 1).alias("early_period_avg"))
)

df_late = (
    df_cancer
    .filter(F.col("year").between(2016, 2022))
    .filter(F.col("has_incidence_data") == True)
    .groupBy("state_abbr", "state_name")
    .agg(F.round(F.avg("cancer_incidence_rate"), 1).alias("late_period_avg"))
)

df_trend = (
    df_early.join(df_late, on=["state_abbr", "state_name"], how="inner")
    .withColumn("rate_change",
        F.round(F.col("late_period_avg") - F.col("early_period_avg"), 1))
    .withColumn("pct_change",
        F.round((F.col("late_period_avg") - F.col("early_period_avg")) /
                F.col("early_period_avg") * 100, 1))
    .withColumn("trend",
        F.when(F.col("rate_change") > 5, "Increasing")
        .when(F.col("rate_change") < -5, "Decreasing")
        .otherwise("Stable"))
    .orderBy("rate_change", ascending=False)
)

display(df_trend)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Trend summary — how many states improving vs worsening

# COMMAND ----------

display(
    df_trend.groupBy("trend")
    .agg(
        F.count("state_abbr").alias("state_count"),
        F.round(F.avg("rate_change"), 1).alias("avg_rate_change"),
        F.collect_list("state_abbr").alias("states")
    )
    .orderBy("avg_rate_change", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Key findings summary

# COMMAND ----------

print("""
KEY FINDINGS — Cancer Trends Analysis (1999-2022)
==================================================

1. NATIONAL TREND: Cancer incidence rates have remained relatively stable
   nationally across 24 years, but with significant state-level variation.

2. PERSISTENT HIGH-BURDEN STATES: Kentucky, Delaware, New Jersey, Connecticut,
   and Maine consistently show the highest cancer incidence rates across the
   full study period — not just recent years.

3. MORTALITY BURDEN: West Virginia (35%), Kentucky (33.7%), and Louisiana (33%)
   show the highest mortality-to-incidence ratios, meaning roughly 1 in 3
   cancer diagnoses results in death — suggesting healthcare access disparities.

4. LOW MORTALITY BURDEN: New York (26.5%), New Jersey (26.9%), and Connecticut
   (27%) show the lowest ratios despite high incidence — better screening and
   treatment access likely driving earlier detection and better outcomes.

5. GEOGRAPHIC PATTERN: High incidence clusters in Northeast (CT, NJ, NY, PA, ME)
   but high mortality clusters in Southeast/Appalachia (WV, KY, MS, TN, LA) —
   two distinct geographic risk patterns with different underlying drivers.
""")