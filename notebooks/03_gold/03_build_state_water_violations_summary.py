# Databricks notebook source
# MAGIC %md
# MAGIC # Gold - State Water Violations Summary
# MAGIC
# MAGIC I'm aggregating health-based water violations to state level for the Gold layer.
# MAGIC I join violations to water systems to get state codes, then aggregate by state and year.
# MAGIC
# MAGIC SDWIS uses mixed state code formats — some systems use 2-letter abbreviations,
# MAGIC others use numeric FIPS codes, and some use EPA region codes. I normalize all of
# MAGIC these to standard 2-letter state abbreviations before aggregating.
# MAGIC
# MAGIC Key metrics:
# MAGIC - Total health-based violations per state per year
# MAGIC - MCL violations (contaminant exceeded limit) vs TT violations (treatment failure)
# MAGIC - Distinct contaminants detected
# MAGIC - Population affected (via water system population served)
# MAGIC - Enforcement rate
# MAGIC
# MAGIC **Source tables:** `silver.sdwa_violations`, `silver.sdwa_water_systems`
# MAGIC **Target table:** `gold.state_water_violations_summary`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver tables

# COMMAND ----------

from pyspark.sql import functions as F

df_viol = spark.table("cancer_environment_lakehouse.silver.sdwa_violations")
df_systems = spark.table("cancer_environment_lakehouse.silver.sdwa_water_systems")

print(f"Violations: {df_viol.count():,}")
print(f"Water systems: {df_systems.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build FIPS-to-abbreviation lookup
# MAGIC
# MAGIC SDWIS stores state codes inconsistently — some records use 2-letter abbreviations,
# MAGIC others use zero-padded numeric FIPS codes. I normalize everything to 2-letter abbr.

# COMMAND ----------

fips_to_abbr = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR", "78": "VI"
}

fips_rows = [(k, v) for k, v in fips_to_abbr.items()]
df_fips = spark.createDataFrame(fips_rows, ["fips_code", "state_abbr_lookup"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Normalize state codes in water systems table

# COMMAND ----------

df_systems_normalized = (
    df_systems
    .select("pwsid", "state_code", "population_served_count", "pws_type_code")
    # Join to FIPS lookup
    .join(F.broadcast(df_fips),
          df_systems["state_code"] == df_fips["fips_code"],
          how="left")
    # Use abbreviation if FIPS matched, otherwise keep original state_code
    .withColumn("state_abbr",
        F.when(F.col("state_abbr_lookup").isNotNull(), F.col("state_abbr_lookup"))
        .otherwise(F.col("state_code")))
    .drop("fips_code", "state_abbr_lookup", "state_code")
)

# Confirm state code distribution after normalization
display(
    df_systems_normalized
    .groupBy("state_abbr")
    .count()
    .orderBy(F.desc("count"))
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Filter to valid US states and DC only
# MAGIC
# MAGIC Drop territories, EPA region codes, and any remaining non-standard codes.

# COMMAND ----------

valid_states = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
    "VT","VA","WA","WV","WI","WY"
]

df_systems_clean = df_systems_normalized.filter(
    F.col("state_abbr").isin(valid_states)
)

print(f"Systems with valid state codes: {df_systems_clean.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Join violations to normalized water systems

# COMMAND ----------

df_joined = (
    df_viol
    .join(df_systems_clean, on="pwsid", how="inner")  # inner join — drop unmatched
)

print(f"Matched violations: {df_joined.count():,}")
unmatched = df_viol.count() - df_joined.count()
print(f"Unmatched violations (excluded): {unmatched:,}")

# Confirm state codes look correct
display(
    df_joined.groupBy("state_abbr").count().orderBy(F.desc("count")).limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Aggregate to state-year level

# COMMAND ----------

df_gold = (
    df_joined
    .filter(F.col("violation_year").isNotNull())
    .filter(F.col("violation_year").between(1990, 2024))
    .groupBy("state_abbr", "violation_year")
    .agg(
        F.count("violation_id").alias("total_health_violations"),
        F.sum(F.when(F.col("violation_category_code") == "MCL", 1).otherwise(0))
            .alias("mcl_violations"),
        F.sum(F.when(F.col("violation_category_code") == "TT", 1).otherwise(0))
            .alias("tt_violations"),
        F.sum(F.when(F.col("is_major_violation") == True, 1).otherwise(0))
            .alias("major_violations"),
        F.countDistinct("contaminant_code").alias("distinct_contaminants"),
        F.sum(F.when(F.col("has_enforcement") == True, 1).otherwise(0))
            .alias("enforced_violations"),
        F.round(
            F.sum(F.when(F.col("has_enforcement") == True, 1).otherwise(0)) * 100.0 /
            F.count("violation_id"), 1
        ).alias("enforcement_rate_pct"),
        F.countDistinct("pwsid").alias("affected_systems"),
        F.sum("population_served_count").alias("total_population_affected"),
        F.sum(F.when(F.col("public_notification_tier") == 1, 1).otherwise(0))
            .alias("tier1_notifications")
    )
    .withColumnRenamed("violation_year", "year")
    .orderBy("state_abbr", "year")
)

print(f"Gold rows: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

display(df_gold.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write to Gold

# COMMAND ----------

target_table = gold_table("state_water_violations_summary")

df_gold.write.format("delta").mode("overwrite").saveAsTable(target_table)
print(f"Written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_rows,
# MAGIC     COUNT(DISTINCT state_abbr) AS states,
# MAGIC     MIN(year) AS earliest_year,
# MAGIC     MAX(year) AS latest_year,
# MAGIC     SUM(total_health_violations) AS total_violations,
# MAGIC     SUM(mcl_violations) AS total_mcl,
# MAGIC     SUM(tt_violations) AS total_tt
# MAGIC FROM cancer_environment_lakehouse.gold.state_water_violations_summary;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Top 10 states by total health-based violations (all years)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     state_abbr,
# MAGIC     SUM(total_health_violations) AS total_violations,
# MAGIC     SUM(mcl_violations) AS mcl_violations,
# MAGIC     ROUND(AVG(distinct_contaminants), 1) AS avg_contaminants_per_year,
# MAGIC     ROUND(AVG(enforcement_rate_pct), 1) AS avg_enforcement_rate_pct
# MAGIC FROM cancer_environment_lakehouse.gold.state_water_violations_summary
# MAGIC GROUP BY state_abbr
# MAGIC ORDER BY total_violations DESC
# MAGIC LIMIT 10;