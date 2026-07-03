# Cancer & Environment Lakehouse

**Author:** Naseer Aryobee — Senior Data Engineer  
**Tech:** Python · Apache Spark · Delta Lake · Databricks · MLflow · Tableau Public  
**Status:** 🚧 In Progress

---

## Project Overview

An end-to-end Databricks Lakehouse project that investigates potential associations between cancer incidence rates across the United States and a broad range of contributing environmental, lifestyle, socioeconomic, and water quality factors.

I re-ingested all datasets from my [MySQL DBA predecessor project](https://github.com/aryobeen007/mysql-dba-project) into a modern Lakehouse architecture using Apache Spark and Delta Lake, where they undergo scalable data engineering, transformation, enrichment, and advanced analytics.

> **Note:** This project surfaces statistical associations and correlations. It does not claim causal conclusions without appropriate scientific evidence.

---

## Business Questions

- Which states have the highest cancer incidence rates, and do they correlate with air quality scores?
- Is there a measurable relationship between health-based drinking water violations and cancer rates by state?
- Do states with more CAFO facilities near impaired waterways show higher cancer mortality?
- What environmental and lifestyle factors are the strongest predictors of elevated cancer rates?
- How has cancer incidence trended over time relative to changes in air quality (2000–2022)?

---

## Architecture

I built this project on the **Medallion Architecture** (Bronze → Silver → Gold) on Databricks Free Edition using Unity Catalog and managed Volumes for storage — no cloud billing attached.

```
Raw CSVs (Local)
      │
      ▼
Unity Catalog Volume
  (raw.landing)
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   BRONZE    │ →  │   SILVER    │ →  │    GOLD     │
│  Raw Delta  │    │Cleaned Delta│    │ Aggregated  │
│  33 tables  │    │   Tables    │    │   Tables    │
│  52.8M rows │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                        Analytics &                  Tableau
                        ML (MLflow)                Dashboards
```

---

## Source Datasets

I ingested data from 5 federal agencies across 33 Bronze Delta tables:

| # | Dataset | Source | Bronze Rows | Key Purpose |
|---|---------|--------|-------------|-------------|
| 1 | Cancer Incidence by State (1999–2022) | CDC WONDER | 1,307 | Primary outcome variable |
| 2 | Cancer Mortality by State (2018–2023) | CDC WONDER | 394 | Secondary outcome variable |
| 3 | EPA Air Quality Index (2000–2022) | EPA AQS | 24,488 | Air pollution exposure (23 annual files) |
| 4 | CDC Chronic Disease Indicators | CDC BRFSS | 398,793 | Lifestyle & behavioral risk factors |
| 5 | SDWIS Water Violations & Enforcement | EPA SDWIS | 15,298,031 | Drinking water health violations |
| 6 | SDWIS Public Water Systems | EPA SDWIS | 433,698 | Water system inventory |
| 7 | SDWIS Facilities, Site Visits & More | EPA SDWIS | 6,696,727 | 9 additional SDWIS tables |
| 8 | NPDES/CAFO Permits & Violations | EPA ECHO | 22,959,739 | 15 NPDES tables — CAFO permits, inspections, enforcement |
| 9 | USDA Food Environment Atlas | USDA ERS | 957,753 | Food access & insecurity indicators |
| 10 | USDA Census of Agriculture | USDA NASS | 6,077,214 | Livestock & agricultural intensity |

**Total Bronze:** 52,848,144 rows across 33 Delta tables

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo & folder structure setup | ✅ Complete |
| 1 | Workspace setup & Unity Catalog | ✅ Complete |
| 2 | Bronze layer — raw ingestion (8 notebooks, 33 tables) | ✅ Complete |
| 3 | Silver layer — cleaning & transformation (8 notebooks) | ✅ Complete |
| 4 | Gold layer — aggregated business tables (7 notebooks) | ✅ Complete |
| 5 | Analytics & Machine Learning | 🔄 In Progress |
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
│   ├── 00_setup/          # Workspace config, Unity Catalog schema creation
│   ├── 01_bronze/         # Raw CSV ingestion → 33 Bronze Delta tables
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

- **Storage:** Unity Catalog managed Volumes (`raw.landing`) — free, no cloud billing
- **Format:** Delta Lake throughout all three layers
- **Catalog:** `cancer_environment_lakehouse` with `raw`, `bronze`, `silver`, `gold` schemas
- **Column sanitization:** Generic snake_case sanitizer applied at Bronze ingestion across all datasets
- **Metadata:** Every Bronze row stamped with `ingested_at` and `source_file` for lineage tracing
- **ML Tracking:** MLflow (built into Databricks) for experiment tracking and model comparison
- **BI Layer:** Tableau Public connected to Databricks SQL Warehouse

---

## Predecessor Project

This project builds on my [MySQL DBA End-to-End Project](https://github.com/aryobeen007/mysql-dba-project), where I designed a 4.57 GB star-schema data warehouse (`cancer_environment_db`) with 22.7 million rows across 14 tables, covering the full DBA lifecycle — schema design, ETL, performance optimization, backup & recovery, and role-based access control.
