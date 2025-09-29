#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import os, json, argparse, warnings
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Optional bonus models (imported only if --bonus is passed to keep startup fast)
def _import_bonus():
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier


RANDOM_STATE = 42


# ---------- Utilities ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["0", "1"])
    plt.yticks(ticks, ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_class_balance(y: np.ndarray, out_path: str):
    vals, counts = np.unique(y, return_counts=True)
    plt.figure()
    plt.bar(vals, counts)
    plt.title("Class Balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str], out_dir: str, bins: int = 20) -> List[str]:
    ensure_dir(out_dir)
    saved = []
    for col in numeric_cols:
        values = df[col].dropna()
        if values.empty:
            continue
        plt.figure()
        plt.hist(values, bins=bins, color="#4C72B0", edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"hist_{col}.png")
        plt.savefig(out_path)
        plt.close()
        saved.append(out_path)
    return saved


def plot_numeric_boxplots(df: pd.DataFrame, numeric_cols: List[str], target_col: str, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    saved = []
    if target_col not in df.columns:
        return saved
    target_values = sorted(df[target_col].dropna().unique())
    if not target_values:
        return saved
    for col in numeric_cols:
        groups = [df.loc[df[target_col] == tv, col].dropna() for tv in target_values]
        if not any(len(g) for g in groups):
            continue
        plt.figure()
        plt.boxplot(groups, labels=[str(tv) for tv in target_values], patch_artist=True)
        plt.title(f"{col} by {target_col}")
        plt.xlabel(target_col)
        plt.ylabel(col)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"boxplot_{col}.png")
        plt.savefig(out_path)
        plt.close()
        saved.append(out_path)
    return saved


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], target_col: str, out_path: str):
    corr_cols = list(numeric_cols)
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        corr_cols = corr_cols + [target_col]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_cols = list(dict.fromkeys(corr_cols))
    if len(corr_cols) < 2:
        return None
    corr = df[corr_cols].corr().fillna(0)
    plt.figure(figsize=(max(6, len(corr_cols) * 0.6), max(4, len(corr_cols) * 0.6)))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_categorical_crosstabs(df: pd.DataFrame, categorical_cols: List[str], target_col: str, out_path: str) -> str:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    if target_col not in df.columns or not categorical_cols:
        save_json(result, out_path)
        return out_path

    for col in categorical_cols:
        counts_df = pd.crosstab(df[col], df[target_col], dropna=False)
        proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)

        counts_serialized = {
            str(idx): {str(k): int(v) for k, v in row.items()}
            for idx, row in counts_df.astype(int).iterrows()
        }
        proportions_serialized = {
            str(idx): {str(k): float(v) for k, v in row.items()}
            for idx, row in proportions_df.iterrows()
        }

        result[col] = {
            "counts": counts_serialized,
            "row_proportions": proportions_serialized,
        }

    save_json(result, out_path)
    return out_path


def groupwise_impute(df: pd.DataFrame, group_cols: List[str], numeric_cols: List[str], categorical_cols: List[str]) -> None:
    if not numeric_cols and not categorical_cols:
        return

    valid_group_cols = [c for c in group_cols if c in df.columns]
    grouped = df.groupby(valid_group_cols) if valid_group_cols else None

    def _fill_numeric(series: pd.Series) -> pd.Series:
        if series.isna().sum() == 0:
            return series
        median = series.median()
        if pd.isna(median):
            return series
        return series.fillna(median)

    def _fill_mode(series: pd.Series) -> pd.Series:
        if series.isna().sum() == 0:
            return series
        mode = series.mode(dropna=True)
        if not mode.empty:
            return series.fillna(mode.iloc[0])
        non_na = series.dropna()
        if not non_na.empty:
            return series.fillna(non_na.iloc[0])
        return series

    if grouped is not None:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = grouped[col].transform(_fill_numeric)
        for col in categorical_cols:
            if col in df.columns:
                df[col] = grouped[col].transform(_fill_mode)

    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            if not pd.isna(median):
                df[col].fillna(median, inplace=True)

    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col].fillna(mode.iloc[0], inplace=True)
            else:
                non_na = df[col].dropna()
                if not non_na.empty:
                    df[col].fillna(non_na.iloc[0], inplace=True)

    remaining_cols = [c for c in dict.fromkeys(numeric_cols + categorical_cols) if c in df.columns]
    if remaining_cols:
        df.dropna(subset=remaining_cols, inplace=True)


# ---------- Pipeline Steps ----------
def load_and_prepare(path: str) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """
    Loads CSV and applies Step 1 cleaning rules:
    - Drop slope, ca, thal (exercise simplification)
    - Drop high‑NA columns (>30%), then drop remaining NA rows
    - Remove ID‑like column if present
    - Binarize target: (num > 0) -> 1 else 0
    Returns cleaned df, target column, numeric cols, categorical cols.
    """
    df = pd.read_csv(path)

    for c in ["slope", "ca", "thal"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    na_pct = df.isna().mean() * 100
    high_na_cols = na_pct[na_pct > 30].index.tolist()
    df.drop(columns=high_na_cols, errors="ignore", inplace=True)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)
        target = "target"
    elif "target" in df.columns:
        target = "target"
    else:
        raise ValueError("Expected label column 'num' or 'target'.")

    feature_cols = [c for c in df.columns if c != target]
    numeric_initial = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    bool_cols = [c for c in numeric_initial if pd.api.types.is_bool_dtype(df[c])]
    categorical_initial = [c for c in feature_cols if c not in numeric_initial]

    group_cols = [c for c in ["dataset", "sex"] if c in df.columns]
    numeric_for_impute = [c for c in numeric_initial if c not in bool_cols]
    categorical_for_impute = list(dict.fromkeys(categorical_initial + bool_cols))
    groupwise_impute(df, group_cols, numeric_for_impute, categorical_for_impute)

    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    for c in categorical_cols:
        df[c] = df[c].astype("category")

    return df, target, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_cols),
    ])


def train_logreg_with_grid(x_train, y_train, preprocessor) -> GridSearchCV:
    """
    Lean yet effective grid (fast; adjust as needed).
    """
    pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
    param_grid = {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__C": [0.1, 0.5, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=1, refit=True, verbose=0)  # n_jobs=1 keeps it snappy
    grid.fit(x_train, y_train)
    return grid


def evaluate(model, x_test, y_test) -> tuple[dict[str, float | dict[str, float] | str | dict], Any]:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "class0": {
            "precision": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        },
        "class1": {
            "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def run_bonus_suite(x_train, x_test, y_train, y_test, preprocessor) -> Dict:
    svc, k_neighbors_classifier, decision_tree_classifier, random_forest_classifier, gradient_boosting_classifier = _import_bonus()

    configs = {
        "SVM": (Pipeline([("pre", preprocessor), ("clf", svc())]), {"clf__C": [0.5, 1.0], "clf__kernel": ["rbf", "linear"]}),
        "KNN": (Pipeline([("pre", preprocessor), ("clf", k_neighbors_classifier())]), {"clf__n_neighbors": [3, 5]}),
        "DecisionTree": (Pipeline([("pre", preprocessor), ("clf", decision_tree_classifier(random_state=RANDOM_STATE))]), {"clf__max_depth": [None, 5, 10]}),
        "RandomForest": (Pipeline([("pre", preprocessor), ("clf", random_forest_classifier(random_state=RANDOM_STATE))]), {"clf__n_estimators": [100, 300]}),
        "GradientBoosting": (Pipeline([("pre", preprocessor), ("clf", gradient_boosting_classifier(random_state=RANDOM_STATE))]), {"clf__n_estimators": [100], "clf__learning_rate": [0.1]}),
    }
    out = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, (pipe, grid_params) in configs.items():
        grid = GridSearchCV(pipe, grid_params, scoring="f1", cv=cv, n_jobs=1, refit=True, verbose=0)
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)
        out[name] = {
            "best_params": grid.best_params_,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_pos": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        }
    return out


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "heart_disease_uci.csv"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--bonus", action="store_true", help="Run bonus model comparisons")
    args = parser.parse_args()

    ensure_dir(args.out)
    warnings.filterwarnings("ignore")

    # Load & clean
    df, target, numeric_cols, categorical_cols = load_and_prepare(args.data)
    x = df[[c for c in df.columns if c != target]]
    y = df[target].values

    # EDA artifacts
    class_balance_path = os.path.join(args.out, "class_balance.png")
    plot_class_balance(y, class_balance_path)
    skewness = df[numeric_cols].skew(numeric_only=True).to_dict()

    eda_dir = os.path.join(args.out, "eda")
    ensure_dir(eda_dir)
    histogram_paths = plot_numeric_histograms(df, numeric_cols, eda_dir)
    boxplot_paths = plot_numeric_boxplots(df, numeric_cols, target, eda_dir)
    correlation_heatmap_path = plot_correlation_heatmap(df, numeric_cols, target, os.path.join(eda_dir, "correlation_heatmap.png"))
    crosstab_path = save_categorical_crosstabs(df, categorical_cols, target, os.path.join(eda_dir, "categorical_target_crosstabs.json"))

    # Preprocess
    pre = build_preprocessor(numeric_cols, categorical_cols)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Train + tune Logistic Regression
    grid = train_logreg_with_grid(x_train, y_train, pre)
    best = grid.best_estimator_

    # Evaluate
    metrics, cm = evaluate(best, x_test, y_test)
    confusion_matrix_path = os.path.join(args.out, "confusion_matrix_logreg.png")
    plot_confusion_matrix(cm, confusion_matrix_path)

    # Optional bonus
    bonus = {}
    if args.bonus:
        bonus = run_bonus_suite(x_train, x_test, y_train, y_test, pre)

    # Save summary
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
    print("Done. Artifacts saved to:", args.out)


if __name__ == "__main__":
    main()
