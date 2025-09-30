# Credit Card Fraud Detection Hackathon Pipeline

This folder contains a ready-to-run analytics and machine-learning pipeline for the credit-card fraud detection hackathon. The workflow covers exploratory data analysis, feature engineering, model training, evaluation, and exports tailored for dashboarding tools such as Tableau or PowerBI.

## 1. Environment Setup

1. Use Python 3.10+.
2. Install the core libraries (create a virtual environment if desired):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly plotnine xgboost
```

> `plotly`, `plotnine`, `imbalanced-learn`, and `xgboost` are optional. The script detects their absence and skips the related functionality. Install them to unlock the full experience.

## 2. Dataset

Place the Kaggle `creditcard.csv` file inside this directory (already provided here). The file must include the `Class` column where 1 denotes fraud and 0 denotes legitimate transactions.

## 3. Running the Pipeline

Execute the script from this directory (or adjust `--data-path` accordingly):

```bash
python credit_fraud_pipeline.py --data-path creditcard.csv --output-dir outputs --use-smote --include-xgboost
```

Useful flags:

- `--skip-eda`: skips heavy visualisations when you only want model results.
- `--train-sample-size N`: subsample the dataset to `N` rows for rapid experiments.
- `--max-eda-sample N`: control how many rows are used for plots (default 60k).
- `--use-smote`: enables SMOTE oversampling (requires `imbalanced-learn`).
- `--include-xgboost`: trains an additional gradient boosting model if XGBoost is available.

All artefacts are written to the folder supplied via `--output-dir` (default: `./outputs`). Feel free to delete and rerun as needed.

## 4. Output Artefacts

The pipeline creates the following sub-folders:

- `figures/`: PNG plots (class balance, distributions, ROC curves, confusion matrices, etc.)
- `reports/`: JSON metrics, classification reports, feature importance tables.
- `tables/`: CSV summaries including model metrics.
- `dashboards/`: Aggregations ready for Tableau/PowerBI (`class_summary.csv`, `hourly_trends.csv`, `rolling_fraud_rate.csv`).

Interactive Plotly dashboards are stored as standalone HTML files under `figures/`.

## 5. Suggested Hackathon Deliverables

1. **Notebook or Slides** – combine the generated plots and metrics to narrate your findings.
2. **Dashboard** – connect the CSVs in `outputs/dashboards` to Tableau/PowerBI for live visuals.
3. **Model Card** – capture precision, recall, ROC-AUC, and feature importance insights for stakeholders.
4. **Fraud Prevention Notes** – propose how to productionise the model (see quick ideas below).

## 6. Real-time Fraud Prevention Ideas

- Monitor the rolling fraud rate (`rolling_fraud_rate.csv`) to trigger alerts when the short-term mean spikes above historical baselines.
- Deploy the high-recall model (logistic regression or gradient boosting) behind a REST API that scores incoming transactions in real time.
- Route high-risk transactions for manual review while logging model explanations (feature importances) to support analysts.
- Enrich the data with merchant metadata, geolocation, device fingerprinting, and prior customer behaviour to boost detection accuracy.
- Implement feedback loops where confirmed fraud/legitimate decisions are fed back for periodic re-training.

## 7. Optional Extensions

- **Web Scraping**: Use Selenium to capture merchant risk signals (e.g., complaint counts, Trustpilot scores) and merge with the dataset.
- **Tableau/PowerBI**: Build an interactive dashboard using the exported CSVs, highlighting hourly fraud spikes and top risky feature ranges.
- **Experiment Tracking**: Integrate MLflow or Weights & Biases to log model runs, metrics, and artefacts.

Happy hacking! Document any adjustments you make so the judging panel can reproduce your results.
