from __future__ import annotations

from textwrap import dedent

import numpy as np
import pandas as pd

from .config import PREPROCESSING_DIR, SUMMARIES_DIR
from .io_utils import write_metadata
from .models import PreprocessingArtifacts


def clean_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Basic data cleaning: drop duplicates and constant columns."""
    logger.info("Cleaning dataset: removing duplicates and constant columns")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    # Retrieving columns with only one unique value
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        logger.info("Dropping constant columns: %s", constant_columns)
        df = df.drop(columns=constant_columns)

    # Map Attrition to binary (Yes = 1, No = 0)
    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    metadata = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "constant_columns_removed": constant_columns,
    }
    write_metadata(metadata, "metadata.json", logger=logger)

    return df


def summarize_dataset(df: pd.DataFrame, logger) -> None:
    """Persist high-level dataset summaries to the outputs folder."""
    logger.info("Generating dataset summaries")
    numeric_summary = df.describe().transpose()
    categorical_summary = df.select_dtypes(include="object").describe().transpose()
    attrition_counts = df["Attrition"].value_counts().rename_axis("Attrition").to_frame("Count")
    attrition_rate = df["Attrition"].value_counts(normalize=True).mul(100).round(2)

    numeric_path = SUMMARIES_DIR / "numeric_summary.csv"
    categorical_path = SUMMARIES_DIR / "categorical_summary.csv"
    distribution_path = SUMMARIES_DIR / "attrition_distribution.csv"

    numeric_summary.to_csv(numeric_path)
    categorical_summary.to_csv(categorical_path)
    attrition_counts.to_csv(distribution_path)

    logger.info("Saved numeric summary to %s", numeric_path)
    logger.info("Saved categorical summary to %s", categorical_path)
    logger.info("Saved attrition distribution to %s", distribution_path)

    quick_facts_path = SUMMARIES_DIR / "quick_facts.txt"
    quick_facts_path.write_text(
        dedent(
            f"""
            Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
            Overall attrition rate: {attrition_rate.get('Yes', 0):.2f}%
            """
        ).strip(),
        encoding="utf-8",
    )
    logger.info("Wrote quick facts to %s", quick_facts_path)


def preprocess_features(df: pd.DataFrame, logger) -> PreprocessingArtifacts:
    """Generate standardized, normalized, and encoded feature sets."""
    logger.info("Preprocessing numeric and categorical features")
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != "AttritionFlag"]
    categorical_columns = [col for col in df.select_dtypes(include="object").columns if col != "Attrition"]

    numeric_frame = df[numeric_columns].copy()
    standardized = (numeric_frame - numeric_frame.mean()) / numeric_frame.std(ddof=0)
    standardized = standardized.add_suffix("_zscore")

    normalized = (numeric_frame - numeric_frame.min()) / (numeric_frame.max() - numeric_frame.min())
    normalized = normalized.add_suffix("_minmax")

    encoded = pd.get_dummies(df[categorical_columns], drop_first=True)

    standardized_path = PREPROCESSING_DIR / "numeric_standardized.csv"
    normalized_path = PREPROCESSING_DIR / "numeric_normalized.csv"
    encoded_path = PREPROCESSING_DIR / "categorical_encoded.csv"

    standardized.to_csv(standardized_path, index=False)
    normalized.to_csv(normalized_path, index=False)
    encoded.to_csv(encoded_path, index=False)

    logger.info("Saved standardized numerics to %s", standardized_path)
    logger.info("Saved normalized numerics to %s", normalized_path)
    logger.info("Saved encoded categoricals to %s", encoded_path)

    model_ready = pd.concat([
        standardized,
        encoded,
        df[["AttritionFlag"]].reset_index(drop=True),
    ], axis=1)
    model_ready_path = PREPROCESSING_DIR / "model_ready_features.csv"
    model_ready.to_csv(model_ready_path, index=False)
    logger.info("Saved model-ready feature matrix to %s", model_ready_path)

    return PreprocessingArtifacts(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        standardized_frame=standardized,
        normalized_frame=normalized,
        encoded_frame=encoded,
    )
