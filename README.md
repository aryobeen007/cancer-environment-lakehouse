# Cancer & Environment Lakehouse

**Author:** Naseer Aryobee — Senior Data Engineer  
**Tech:** Python · Apache Spark · Delta Lake · Databricks · MLflow · Tableau Public  
**Status:** 🚧 In Progress

---

## Project Overview

An end-to-end Databricks Lakehouse project that investigates potential associations between cancer incidence rates across the United States and a broad range of contributing environmental, lifestyle, socioeconomic, and water quality factors.

This project re-ingests all datasets from the [MySQL DBA predecessor project](https://github.com/aryobeen007/mysql-dba-project) into a modern Lakehouse architecture using Apache Spark and Delta Lake, where they undergo scalable data engineering, transformation, enrichment, and advanced analytics.

> **Note:** This project surfaces statistical associations and correlations. It does not claim causal conclusions without appropriate scientific evidence.

---

## Business Questions

- Which states have the highest cancer incidence rates, and do they correlate with air quality scores?
- Is there a measurable relationship between health-based drinking water violations and cancer rates by state?
- Do states with more CAFO facilities near impaired waterways show higher cancer mortality?
- What environmental and lifestyle factors are the strongest predictors of elevated cancer rates?
- How has cancer incidence trended over time relative to changes in air quality (2000–2023)?

---

## Architecture

This project follows the **Medallion Architecture** (Bronze → Silver → Gold) on Databricks Community Edition with DBFS storage.

```
Raw CSVs (Local)
      │
      ▼
  DBFS Upload
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   BRONZE    │ →  │   SILVER    │ →  │    GOLD     │
│  Raw Delta  │    │Cleaned Delta│    │ Aggregated  │
│   Tables    │    │   Tables    │    │   Tables    │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                        Analytics &                  Tableau
                        ML (MLflow)                Dashboards
```

---

## Source Datasets

| # | Dataset | Source | Rows | Key Purpose |
|---|---------|--------|------|-------------|
| 1 | Cancer Incidence by State | CDC / NCI SEER | 1,218 | Primary outcome variable |
| 2 | Cancer Mortality by State | CDC / NCI SEER | 306 | Secondary outcome variable |
| 3 | EPA Air Quality Index (2000–2023) | EPA AQS | 23,100 | Air pollution exposure |
| 4 | EPA SDWIS Water Violations | EPA SDWIS | 15,298,031 | Drinking water health violations |
| 5 | EPA SDWIS Water System Inventory | EPA SDWIS | 433,698 | Geographic water system lookup |
| 6 | USDA Food Environment Atlas | USDA ERS | 930,317 | Food access & insecurity indicators |
| 7 | CDC Chronic Disease Indicators (BRFSS) | CDC BRFSS | 375,987 | Lifestyle & behavioral risk factors |
| 8 | USDA Census of Agriculture | USDA NASS | 3,339,228 | Livestock & agricultural intensity |
| 9 | EPA ECHO CAFO Facilities | EPA ECHO | 1,188,507 | CAFOs near impaired waterways |

**Total:** ~22.7 million rows across 9 source datasets

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo & folder structure setup | ✅ Complete |
| 1 | Workspace setup & DBFS upload | 🔄 In Progress |
| 2 | Bronze layer — raw ingestion (9 notebooks) | ⏳ Pending |
| 3 | Silver layer — cleaning & transformation (9 notebooks) | ⏳ Pending |
| 4 | Gold layer — aggregated business tables (7 notebooks) | ⏳ Pending |
| 5 | Analytics & Machine Learning | ⏳ Pending |
| 6 | Tableau dashboards & portfolio | ⏳ Pending |

---

## Tableau Dashboards

*Links will be added as dashboards are published.*

- Dashboard 1 — Cancer & Environment Overview *(coming soon)*
- Dashboard 2 — Water Quality Deep Dive *(coming soon)*
- Dashboard 3 — Environmental Risk Profile *(coming soon)*

---

## Repository Structure

```
cancer-environment-lakehouse/
├── notebooks/
│   ├── 00_setup/          # Workspace config, schema creation
│   ├── 01_bronze/         # Raw CSV ingestion → Delta tables
│   ├── 02_silver/         # Cleaning, typing, deduplication
│   ├── 03_gold/           # Aggregated business-ready tables
│   ├── 04_analytics/      # Correlation & trend analysis
│   ├── 05_ml/             # Feature engineering & ML models
│   └── 06_workflows/      # Pipeline orchestration
├── etl/                   # Standalone Python ETL/upload scripts
├── sql/
│   ├── analytics/         # Reference Spark SQL queries
│   └── validation/        # Data quality validation queries
├── data/tableau/          # CSV exports for Tableau connection
├── tableau/               # Tableau workbook files
├── diagrams/              # Architecture & data flow diagrams
├── docs/                  # Data dictionary & project documentation
├── screenshots/           # Databricks UI, MLflow, dashboard previews
├── .gitignore
└── README.md
```

---

## Key Technical Decisions

- **Storage:** DBFS (Databricks File System) — free tier, no cloud billing
- **Format:** Delta Lake throughout all three layers
- **Partitioning:** Large tables partitioned by `state_code`/`state_fips` for query performance
- **Optimization:** Z-ordering on high-cardinality filter columns; broadcast joins for small dimension tables (`dim_state` = 51 rows, `dim_year` = 47 rows)
- **ML Tracking:** MLflow (built into Databricks) for experiment tracking and model comparison
- **BI Layer:** Tableau Public connected to Databricks SQL Warehouse

---

## Predecessor Project

This project builds on the [MySQL DBA End-to-End Project](https://github.com/aryobeen007/mysql-dba-project), which constructed a 4.57 GB star-schema data warehouse (`cancer_environment_db`) with 22.7 million rows across 14 tables, covering the full DBA lifecycle — schema design, performance optimization, backup & recovery, and role-based access control.
