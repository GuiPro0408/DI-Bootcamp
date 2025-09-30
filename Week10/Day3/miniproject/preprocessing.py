"""Data loading and preprocessing helpers."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from data_io import load_csv


def _log(step: str, **details: object) -> None:
    """Emit a preprocessing-specific trace message."""
    if details:
        formatted = " | ".join(f"{k}={v}" for k, v in details.items())
        print(f"[PRE] {step:<28} :: {formatted}")
    else:
        print(f"[PRE] {step}")


def groupwise_impute(
        df: pd.DataFrame, group_cols: List[str], numeric_cols: List[str], categorical_cols: List[str]
) -> None:
    """Fill missing numeric and categorical values using grouped statistics.

    The dataframe is updated in place by first applying the median or mode within
    the provided groups and then falling back to global statistics. Any rows that
    still contain missing values in the tracked columns are dropped as a last
    resort.

    Args:
        df: Dataset to modify.
        group_cols: Column names that define imputation groups (e.g., dataset, sex).
        numeric_cols: Numeric feature columns to impute.
        categorical_cols: Categorical or boolean feature columns to impute.
    """
    if not numeric_cols and not categorical_cols:
        return

    valid_group_cols = [c for c in group_cols if c in df.columns]
    grouped = df.groupby(valid_group_cols) if valid_group_cols else None
    _log(
        "Imputation setup",
        groups=valid_group_cols or "none",
        numeric=len(numeric_cols),
        categorical=len(categorical_cols),
    )

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
        before_drop = df[remaining_cols].isna().sum().sum()
        df.dropna(subset=remaining_cols, inplace=True)
        after_drop = df[remaining_cols].isna().sum().sum()
        _log("Rows dropped post-imputation", before=before_drop, after=after_drop, rows=len(df))
    else:
        _log("No monitored columns after imputation")


def load_and_prepare(path: str) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Load the heart-disease dataset, clean columns, and prepare metadata.

    Args:
        path: CSV file path.

    Returns:
        Cleaned dataframe, target column name, numeric feature names, and
        categorical feature names.
    """
    _log("Loading dataset", path=path)
    df = load_csv(path)
    _log("Raw shape", rows=df.shape[0], cols=df.shape[1])

    for c in ["slope", "ca", "thal"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            _log("Dropped column", column=c)

    na_pct = df.isna().mean() * 100
    high_na_cols = na_pct[na_pct > 30].index.tolist()
    df.drop(columns=high_na_cols, errors="ignore", inplace=True)
    if high_na_cols:
        _log("High-NA columns removed", columns=high_na_cols)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
        _log("Dropped column", column="id")

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
    _log(
        "Initial type split",
        numeric=len(numeric_initial),
        categorical=len(categorical_initial),
        bool=len(bool_cols),
    )

    group_cols = [c for c in ["dataset", "sex"] if c in df.columns]
    numeric_for_impute = [c for c in numeric_initial if c not in bool_cols]
    categorical_for_impute = list(dict.fromkeys(categorical_initial + bool_cols))
    groupwise_impute(df, group_cols, numeric_for_impute, categorical_for_impute)

    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    for c in categorical_cols:
        df[c] = df[c].astype("category")

    _log(
        "Prepared dataset",
        rows=df.shape[0],
        cols=len(feature_cols) + 1,
        target=target,
        numeric=len(numeric_cols),
        categorical=len(categorical_cols),
    )
    return df, target, numeric_cols, categorical_cols
