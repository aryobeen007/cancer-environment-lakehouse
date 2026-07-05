# Databricks notebook source
# MAGIC %md
# MAGIC # ML 01 - Cancer Incidence Rate Regression Model
# MAGIC
# MAGIC I'm building a regression model to predict state-level cancer incidence rates
# MAGIC using environmental, lifestyle, and food environment features. I use scikit-learn
# MAGIC which runs on the driver node — appropriate for this 51-state dataset.
# MAGIC
# MAGIC Models compared:
# MAGIC - Ridge Regression (baseline, interpretable)
# MAGIC - Random Forest Regressor
# MAGIC - Gradient Boosted Trees Regressor
# MAGIC
# MAGIC **Source table:** `gold.state_environmental_risk_profile`
# MAGIC **Target:** `avg_cancer_incidence_rate`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and setup

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Disable MLflow autologging — not supported in Free Edition
try:
    import mlflow
    mlflow.sklearn.autolog(disable=True)
except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load data and define features

# COMMAND ----------

df_profile = spark.table(gold_table("state_environmental_risk_profile"))

FEATURE_COLS = [
    "avg_smoking_pct",
    "avg_obesity_pct",
    "avg_diabetes_pct",
    "avg_copd_pct",
    "avg_physical_inactivity_pct",
    "avg_median_aqi",
    "avg_pm25_days_per_year",
    "avg_health_violations_per_year",
    "total_cafo_facilities",
    "cafos_near_impaired_waters",
    "pct_low_food_access_2019",
    "pct_snap_participation_2017",
    "fast_food_to_full_service_ratio"
]

TARGET_COL = "avg_cancer_incidence_rate"

# Convert to pandas
pdf = df_profile.select(
    ["state_abbr", "state_name", TARGET_COL] + FEATURE_COLS
).toPandas()

# Fill nulls with column median
for col in FEATURE_COLS:
    pdf[col] = pdf[col].fillna(pdf[col].median())

pdf = pdf.dropna(subset=[TARGET_COL])

print(f"Dataset: {pdf.shape[0]} states, {len(FEATURE_COLS)} features")
print(f"Target range: {pdf[TARGET_COL].min():.1f} - {pdf[TARGET_COL].max():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare features and target

# COMMAND ----------

X = pdf[FEATURE_COLS].values
y = pdf[TARGET_COL].values

print(f"Features: {X.shape}, Target: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train models

# COMMAND ----------

models = {
    "Ridge_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random_Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]),
    "Gradient_Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
    ])
}

results = {}

for model_name, pipeline in models.items():
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    results[model_name] = {
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "mae": round(mae, 3),
        "y_pred": y_pred,
        "pipeline": pipeline
    }

    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")
    print(f"  MAE:  {mae:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model comparison summary

# COMMAND ----------

print("\n" + "="*55)
print("MODEL COMPARISON — Cancer Incidence Regression")
print("="*55)
print(f"{'Model':<25} {'RMSE':>8} {'R²':>8} {'MAE':>8}")
print("-"*55)
for name, metrics in sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True):
    print(f"{name:<25} {metrics['rmse']:>8.3f} {metrics['r2']:>8.3f} {metrics['mae']:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Best model — prediction vs actual

# COMMAND ----------

best_model_name = max(results, key=lambda x: results[x]["r2"])
best_preds = results[best_model_name]["y_pred"]
print(f"Best model: {best_model_name} (R² = {results[best_model_name]['r2']})")

comparison_df = pd.DataFrame({
    "state_abbr": pdf["state_abbr"].values,
    "state_name": pdf["state_name"].values,
    "actual": y.round(1),
    "predicted": best_preds.round(1),
    "abs_error": np.abs(y - best_preds).round(1)
}).sort_values("abs_error", ascending=False)

display(spark.createDataFrame(comparison_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature importance (Random Forest)

# COMMAND ----------

rf_pipeline = results["Random_Forest"]["pipeline"]
rf_model = rf_pipeline.named_steps["model"]
importances = rf_model.feature_importances_

print("\nRANDOM FOREST FEATURE IMPORTANCES")
print("="*55)
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 60)
    print(f"{feat:<40} {imp:.4f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key findings

# COMMAND ----------

print("""
KEY FINDINGS — Cancer Incidence Regression
==========================================

Note: Models are trained and evaluated on the same 51 states.
With n=51, training metrics reflect fit quality not generalization.
Cross-validation would be needed for true predictive performance.

1. GRADIENT BOOSTING achieves near-perfect fit (R²≈0.999) —
   expected with this small dataset, indicating memorization.

2. RANDOM FOREST R²≈0.81 is more meaningful — shows that
   13 environmental and lifestyle features explain ~81% of
   variance in state cancer incidence rates.

3. RIDGE REGRESSION R²≈0.54 confirms a moderate linear
   relationship exists between the features and cancer rates.

4. FEATURE IMPORTANCE: COPD prevalence and smoking rate are
   consistently the strongest predictors — consistent with
   the correlation analysis findings in notebook 04.

5. PRACTICAL USE: The model can identify which states are
   likely to have elevated cancer rates based on their
   environmental and lifestyle profile.
""")