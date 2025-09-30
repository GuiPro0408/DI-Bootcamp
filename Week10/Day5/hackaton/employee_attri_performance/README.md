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
The script loads `WA_Fn-UseC_-HR-Employee-Attrition.csv`, performs cleaning,
EDA, feature scaling (standardization + min-max normalization), categorical
encoding, correlation analysis, and generates visuals + summary artifacts.

## Generated Outputs
All outputs are written to the `outputs/` folder:
- `numeric_summary.csv`, `categorical_summary.csv`, `attrition_distribution.csv`
- `numeric_standardized.csv`, `numeric_normalized.csv`, `categorical_encoded.csv`
- `model_ready_features.csv` for downstream modelling
- Aggregated insights such as `attrition_rate_by_jobrole.csv`
- Visualization files (PNG) covering attrition distributions, heatmaps, etc.
- `retention_recommendations.txt` summarising actionable takeaways
- (Optional) `plotly_attrition_age.html`, `plotnine_*.png` when the respective
  libraries are installed

## Next Steps
1. Review `retention_recommendations.txt` to shape your hackathon storyline.
2. Pull CSV summaries/visuals into Tableau or PowerBI for dashboarding.
3. Optionally extend the script with predictive modelling or interactive apps.
