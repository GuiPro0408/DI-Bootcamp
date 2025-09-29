# Heart Disease Prediction Mini-Project

This mini-project walks through a complete applied machine-learning workflow for predicting the presence of heart
disease using data from the UCI repository. It combines lightweight exploratory data analysis (EDA), preprocessing,
model tuning, evaluation, and optional model comparisons — all in a single, reproducible Python script.

## Key Features

- **Data cleaning**: Drops high-missing columns, removes ID-like columns, and binarises the `num` target into a `target`
  label (0 = no disease, 1 = disease).
- **Exploratory analysis**:
    - Class balance bar chart.
    - Numeric histograms and boxplots (per target class).
    - Numeric correlation heatmap.
    - Categorical cross-tabulations against the target, saved as JSON.
- **Preprocessing**: Min–Max scaling for numeric features and one-hot encoding for categorical variables.
- **Modeling**: Logistic Regression tuned via `GridSearchCV` with stratified folds.
- **Evaluation**: Accuracy, precision, recall, F1 (per class), confusion matrix plot, and a consolidated JSON summary.
- **Optional bonus**: Compare SVM, KNN, Decision Tree, Random Forest, and Gradient Boosting models with the same
  preprocessing pipeline.

## Project Structure

```
Week10/Day3/miniproject/
├── heart_disease_uci.csv          # Source dataset
├── config.py                      # Shared configuration constants
├── data_io.py                     # File-system and JSON helpers
├── preprocessing.py               # Data loading, cleaning, and imputation
├── modeling.py                    # Preprocessing pipeline, training, evaluation, bonus models
├── eda.py                         # Plotting and categorical cross-tab utilities
├── predicting_heart_disease.py    # Orchestrator CLI tying all components together
└── README.md
```

## Usage

Run the main script to train the logistic regression model and generate EDA artifacts:

```powershell
python predicting_heart_disease.py --out outputs
```

- `--out` controls where artifacts are written (default: `outputs` in the same directory).
- The script uses the bundled `heart_disease_uci.csv` by default. Provide a different CSV via `--data <path>` if
  desired.

### Bonus Model Comparison

To run the optional model comparison suite:

```powershell
python predicting_heart_disease.py --bonus
```

This produces additional performance summaries for SVM, KNN, Decision Tree, Random Forest, and Gradient Boosting
classifiers (with modest hyperparameter grids).

## Generated Artifacts

Running the script creates (or updates) the specified output directory with:

- `summary.json`: Consolidated metrics, selected hyperparameters, skewness statistics, cross-tab paths, and artifact
  locations.
- `class_balance.png`: Bar chart of target class distribution.
- `confusion_matrix_logreg.png`: Confusion matrix heatmap for the logistic regression model.
- `eda/`:
    - `hist_<feature>.png`: Histograms for each numeric feature.
    - `boxplot_<feature>.png`: Boxplots of numeric features split by the target class.
    - `correlation_heatmap.png`: Pearson correlation heatmap for numeric features (and the numeric target).
    - `categorical_target_crosstabs.json`: Counts and row-normalized proportions for each categorical feature vs. the
      target.

If `--bonus` is enabled, summaries for the additional models are included under the `bonus_models` key inside
`summary.json`.

## Customisation Tips

- Adjust the preprocessing or model grid in `predicting_heart_disease.py` to experiment with other feature scalers or
  classifiers.
- Swap in a new dataset by matching the expected column names or adapting the `load_and_prepare` function.
- Enrich the EDA section with additional plots or statistics using the existing helper functions as templates.

## Troubleshooting

- **Module not found**: Confirm dependencies are installed in your active environment (
  `pip install -r requirements.txt`).
- **Matplotlib display backend issues**: Since plots are written directly to files, you do not need an interactive
  backend; running in headless environments is supported.
- **Performance concerns**: The grids are intentionally small. Increase `n_jobs` in the `GridSearchCV` calls or enlarge
  grids only if you have adequate compute resources.

Happy experimenting!
