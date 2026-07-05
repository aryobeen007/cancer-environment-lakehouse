# Databricks notebook source
# MAGIC %md
# MAGIC # ML 02 - High Cancer Incidence State Classifier
# MAGIC
# MAGIC I'm building a binary classifier to predict whether a state is "high cancer
# MAGIC incidence" (avg rate >= 500 per 100k) using scikit-learn.
# MAGIC
# MAGIC Class distribution:
# MAGIC - High incidence (True): 16 states
# MAGIC - Low incidence (False): 35 states
# MAGIC
# MAGIC Models: Logistic Regression, Random Forest, Gradient Boosting
# MAGIC
# MAGIC **Source table:** `gold.state_environmental_risk_profile`
# MAGIC **Target:** `is_high_cancer_incidence`

# COMMAND ----------

# MAGIC %run ../00_setup/00_workspace_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and setup

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Disable MLflow autologging — not supported in Free Edition
try:
    import mlflow
    mlflow.sklearn.autolog(disable=True)
except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load data and check class distribution

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

TARGET_COL = "is_high_cancer_incidence"

pdf = df_profile.select(
    ["state_abbr", "state_name", "avg_cancer_incidence_rate", TARGET_COL] + FEATURE_COLS
).toPandas()

for col in FEATURE_COLS:
    pdf[col] = pdf[col].fillna(pdf[col].median())

pdf = pdf.dropna(subset=[TARGET_COL])
pdf["label"] = pdf[TARGET_COL].astype(int)

high_risk = pdf["label"].sum()
low_risk = (pdf["label"] == 0).sum()
print(f"Total states: {len(pdf)}")
print(f"High incidence states (label=1): {high_risk}")
print(f"Low incidence states (label=0): {low_risk}")

high_states = pdf[pdf["label"] == 1]["state_abbr"].tolist()
print(f"\nHigh incidence states: {high_states}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare features and target

# COMMAND ----------

X = pdf[FEATURE_COLS].values
y = pdf["label"].values

print(f"Features: {X.shape}, Target: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train and evaluate classifiers

# COMMAND ----------

models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, C=0.5, random_state=42))
    ]),
    "Random_Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]),
    "Gradient_Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
    ])
}

results = {}

for model_name, pipeline in models.items():
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    results[model_name] = {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "auc": round(auc, 3),
        "y_pred": y_pred,
        "pipeline": pipeline
    }

    print(f"\n{model_name}:")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model comparison summary

# COMMAND ----------

print("\n" + "="*65)
print("MODEL COMPARISON — High Cancer Incidence Classifier")
print("="*65)
print(f"{'Model':<25} {'AUC':>8} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}")
print("-"*65)
for name, m in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
    print(f"{name:<25} {m['auc']:>8.3f} {m['accuracy']:>10.3f} {m['precision']:>11.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Best model — predictions vs actual

# COMMAND ----------

best_model_name = max(results, key=lambda x: results[x]["auc"])
best_preds = results[best_model_name]["y_pred"]
print(f"Best model: {best_model_name} (AUC = {results[best_model_name]['auc']})")

pred_df = pd.DataFrame({
    "state_abbr": pdf["state_abbr"].values,
    "state_name": pdf["state_name"].values,
    "incidence_rate": pdf["avg_cancer_incidence_rate"].round(1).values,
    "actual_high_risk": y,
    "predicted_high_risk": best_preds,
    "correct": (y == best_preds).astype(int)
}).sort_values("incidence_rate", ascending=False)

display(spark.createDataFrame(pred_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Confusion matrix

# COMMAND ----------

cm = confusion_matrix(y, best_preds)
print("\nCONFUSION MATRIX")
print("="*40)
print(f"                  Predicted Low  Predicted High")
print(f"  Actual Low              {cm[0][0]:>4}            {cm[0][1]:>4}")
print(f"  Actual High             {cm[1][0]:>4}            {cm[1][1]:>4}")
print(f"\nTrue Negatives:  {cm[0][0]}  (correctly predicted low risk)")
print(f"False Positives: {cm[0][1]}  (incorrectly predicted high risk)")
print(f"False Negatives: {cm[1][0]}  (missed actual high risk states)")
print(f"True Positives:  {cm[1][1]}  (correctly predicted high risk)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Feature importance (Random Forest)

# COMMAND ----------

rf_pipeline = results["Random_Forest"]["pipeline"]
rf_model = rf_pipeline.named_steps["model"]
importances = rf_model.feature_importances_

print("\nRANDOM FOREST FEATURE IMPORTANCES — High Risk Classifier")
print("="*58)
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 60)
    print(f"{feat:<40} {imp:.4f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Key findings

# COMMAND ----------

print("""
KEY FINDINGS — High Cancer Incidence Classifier
================================================

Note: Models trained and evaluated on same 51 states.
Training metrics reflect fit quality, not generalization.

1. HIGH INCIDENCE STATES (16 of 51): Predominantly Northeast
   and upper Midwest — KY, DE, NJ, CT, ME, NH, RI, NY, PA,
   WV, LA, MI, IA, MA, IL, FL.

2. BEST CLASSIFIER identifies high-risk states with high
   accuracy — lifestyle factors (COPD, smoking, obesity)
   are the strongest discriminating features.

3. COPD PREVALENCE is the most important feature for
   distinguishing high vs low cancer incidence states —
   reflecting the compounding effect of smoking-related
   lung disease and cancer risk.

4. GEOGRAPHIC INSIGHT: The Northeast cluster of high-incidence
   states share high healthcare utilization, dense population,
   and historical industrial pollution exposure — factors
   not fully captured in current features.

5. MISCLASSIFIED STATES tend to have unusual combinations
   of risk factors — high lifestyle burden but low incidence
   (Southeast) or low lifestyle burden but high incidence
   (some Northeast states).
""")