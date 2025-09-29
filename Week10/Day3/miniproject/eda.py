"""Exploratory data analysis helpers for the heart disease project."""

from __future__ import annotations

import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import ensure_dir, save_json


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> str:
    """Render and save a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix values.
        out_path: File path where the PNG should be written.

    Returns:
        The path to the saved image.
    """
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
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_class_balance(y: np.ndarray, out_path: str) -> str:
    """Plot the distribution of binary classes and save the figure.

    Args:
        y: Array of class labels.
        out_path: Destination file path.

    Returns:
        The path to the saved bar chart.
    """
    vals, counts = np.unique(y, return_counts=True)
    plt.figure()
    plt.bar(vals, counts)
    plt.title("Class Balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_numeric_histograms(
    df: pd.DataFrame, numeric_cols: List[str], out_dir: str, bins: int = 20
) -> List[str]:
    """Create histograms for each numeric feature and save them under a directory.

    Args:
        df: Source dataframe.
        numeric_cols: Numeric feature names to plot.
        out_dir: Directory to store the images.
        bins: Number of bins to use per histogram.

    Returns:
        A list with the paths to the generated histogram files.
    """
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


def plot_numeric_boxplots(
    df: pd.DataFrame, numeric_cols: List[str], target_col: str, out_dir: str
) -> List[str]:
    """Generate boxplots of numeric features segmented by the target column.

    Args:
        df: Dataset containing the features and target.
        numeric_cols: Numeric columns to visualise.
        target_col: Target column name used for grouping.
        out_dir: Directory to store boxplot images.

    Returns:
        Paths to the saved boxplot files.
    """
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


def plot_correlation_heatmap(
    df: pd.DataFrame, numeric_cols: List[str], target_col: str, out_path: str
) -> str | None:
    """Save a correlation heatmap for numeric features and optional target column.

    Args:
        df: Dataframe to analyse.
        numeric_cols: Numeric column names used for correlations.
        target_col: Target column to include when numeric.
        out_path: File path to the saved heatmap.

    Returns:
        The path to the heatmap image, or None if insufficient columns were available.
    """
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
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_categorical_crosstabs(
    df: pd.DataFrame, categorical_cols: List[str], target_col: str, out_path: str
) -> str:
    """Persist count and proportion tables for categorical features versus the target.

    Args:
        df: Source dataframe.
        categorical_cols: Categorical feature names to tabulate.
        target_col: Target column name.
        out_path: JSON file path for the serialized output.

    Returns:
        The JSON file path containing counts and row-wise proportions per feature.
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    if target_col not in df.columns or not categorical_cols:
        save_json(result, out_path)
        return out_path

    for col in categorical_cols:
        counts_df = pd.crosstab(df[col], df[target_col], dropna=False)
        proportions_df = (
            counts_df.div(counts_df.sum(axis=1), axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

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
