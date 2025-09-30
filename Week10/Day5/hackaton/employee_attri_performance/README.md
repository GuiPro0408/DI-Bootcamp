# Employee Attrition & Performance Analysis

This folder contains a ready-to-run analytics pipeline for the IBM HR Analytics
Employee Attrition dataset used in Hackathon Subject 3.

## Prerequisites

- Python 3.10+
- Required packages: pandas, numpy, matplotlib
- Optional (enables richer outputs): seaborn, scipy, plotly, plotnine

If `pip` is available, you can install everything with:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy plotly plotnine
```

## How to Run

```bash
python employee_attrition_analysis.py
```

You can optionally provide a custom dataset path:

```bash
python employee_attrition_analysis.py path/to/alternative_dataset.csv
```

Behind the scenes the command delegates to the `run_pipeline` function defined
in `pipeline.py`, ensuring the orchestration layer stays independent from the
analytics components.

The package loads `WA_Fn-UseC_-HR-Employee-Attrition.csv`, performs cleaning,
EDA, feature scaling, categorical encoding, correlation analysis, and generates
visuals + summary artifacts. A detailed execution log is streamed to the console
and stored at `outputs/logs/pipeline.log` for traceability.

## Project Structure

The codebase is organised for separation of concerns:

- `pipeline.py` – orchestration logic that wires all stages together
- `data_processing.py` – data cleaning, summaries, feature engineering
- `analysis.py` – statistical tests and correlation studies
- `visualization.py` – Matplotlib/Seaborn/Plotly/Plotnine chart generation
- `reporting.py` – narrative retention recommendations
- `io_utils.py` – filesystem helpers and dataset loading
- `logging_utils.py` – shared logging configuration
- `config.py` – path constants and directory registry
- `models.py` – shared dataclasses used across stages

Each module can be imported independently (e.g. for notebooks or unit tests),
while `employee_attrition_analysis.py` remains a minimal CLI wrapper.

## Output Structure

All deliverables are written to the `outputs/` directory, organised as follows:

- `summaries/` – numeric & categorical descriptive statistics, attrition counts, quick facts
- `preprocessing/` – standardized & min–max scaled numerics, encoded categoricals, model-ready matrix
- `statistics/` – Spearman correlations, chi-square results, residual diagnostics
- `tables/` – aggregated EDA tables for Tableau/PowerBI (job role, department, education, distance)
- `figures/` – PNG visualisations generated with Matplotlib/Seaborn fallbacks (+ Plotnine when available)
- `interactive/` – Plotly HTML dashboards (created when Plotly is installed)
- `reports/` – narrative recommendations and takeaway summaries
- `metadata/` – dataset cleaning metadata
- `logs/` – execution log file capturing the pipeline flow and warnings

## Recommended Next Steps

1. Review `reports/retention_recommendations.txt` alongside the log file for a
   guided storyline.
2. Import CSV tables or PNGs into Tableau/PowerBI to craft the final dashboard.
3. Extend the pipeline with predictive modelling or interactive front-ends if
   you need additional hackathon features.
