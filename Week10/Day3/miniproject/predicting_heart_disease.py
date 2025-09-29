"""
Predicting Heart Disease Using Logistic Regression
=================================================

Implements the instructor steps:
1) Drop slope, ca, thal; discuss imputation vs. safe drops.
2) EDA (minimal in code: class balance; plus saved skewness numbers).
3) Preprocess: Min‑Max scale numeric; One‑Hot encode categorical.
4) Train/Test split; LogisticRegression tuned with GridSearchCV + StratifiedKFold.
5) Evaluate: accuracy, precision, recall, F1 for class 0 and 1; confusion matrix saved.
6) (Bonus, optional via --bonus) Run a quick model comparison suite.

Docs:
- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

from sklearn.model_selection import train_test_split

from config import RANDOM_STATE
from data_io import ensure_dir, save_json
from eda import (
    plot_class_balance,
    plot_numeric_histograms,
    plot_numeric_boxplots,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    save_categorical_crosstabs,
)
from modeling import build_preprocessor, train_logreg_with_grid, evaluate, run_bonus_suite
from preprocessing import load_and_prepare


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(args.out)
    warnings.filterwarnings("ignore")

    df, target, numeric_cols, categorical_cols = load_and_prepare(args.data)
    x = df[[c for c in df.columns if c != target]]
    y = df[target].values

    class_balance_path = plot_class_balance(y, os.path.join(args.out, "class_balance.png"))
    skewness = df[numeric_cols].skew(numeric_only=True).to_dict()

    eda_dir = os.path.join(args.out, "eda")
    ensure_dir(eda_dir)
    histogram_paths = plot_numeric_histograms(df, numeric_cols, eda_dir)
    boxplot_paths = plot_numeric_boxplots(df, numeric_cols, target, eda_dir)
    correlation_heatmap_path = plot_correlation_heatmap(
        df, numeric_cols, target, os.path.join(eda_dir, "correlation_heatmap.png")
    )
    crosstab_path = save_categorical_crosstabs(
        df, categorical_cols, target, os.path.join(eda_dir, "categorical_target_crosstabs.json")
    )

    pre = build_preprocessor(numeric_cols, categorical_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    grid = train_logreg_with_grid(x_train, y_train, pre)
    best = grid.best_estimator_

    metrics, cm = evaluate(best, x_test, y_test)
    confusion_matrix_path = plot_confusion_matrix(cm, os.path.join(args.out, "confusion_matrix_logreg.png"))

    bonus = run_bonus_suite(x_train, x_test, y_train, y_test, pre) if args.bonus else {}

    summary = {
        "data_shape": list(df.shape),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "skewness": {k: float(v) for k, v in skewness.items()},
        "logreg_best_params": grid.best_params_,
        "logreg_cv_best_f1": float(grid.best_score_),
        "test_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "artifacts": {
            "class_balance": class_balance_path,
            "histograms": histogram_paths,
            "boxplots": boxplot_paths,
            "correlation_heatmap": correlation_heatmap_path,
            "categorical_crosstabs": crosstab_path,
            "confusion_matrix_plot": confusion_matrix_path,
        },
        "bonus_models": bonus,
    }

    save_json(summary, os.path.join(args.out, "summary.json"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "heart_disease_uci.csv"),
        help="Path to the heart disease dataset CSV",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to store generated artifacts",
    )
    parser.add_argument("--bonus", action="store_true", help="Run bonus model comparisons")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)
    print("Done. Artifacts saved to:", args.out)


if __name__ == "__main__":
    main()
