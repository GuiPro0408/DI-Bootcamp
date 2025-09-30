from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, cast

from config import RESIDUALS_DIR, STATISTICS_DIR, TABLES_DIR
from models import PreprocessingArtifacts

try:  # Optional dependency
    from scipy import stats

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SCIPY = False
    stats = None


def statistical_tests(df: pd.DataFrame, logger) -> None:
    """Run chi-square tests on selected categorical variables against attrition."""
    logger.info("Running chi-square tests on key categorical features")
    if not HAS_SCIPY or stats is None:
        message = "SciPy not installed. Install scipy to run chi-square tests."
        (STATISTICS_DIR / "chi_square_results.txt").write_text(message, encoding="utf-8")
        logger.warning(message)
        return

    tests = []
    target = df["Attrition"]

    for column in ["OverTime", "JobRole", "Department", "BusinessTravel", "WorkLifeBalance"]:
        contingency = pd.crosstab(df[column], target)
        scipy_stats = cast(Any, stats)
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        tests.append(
            {
                "feature": column,
                "chi2": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant_at_0.05": bool(p_value < 0.05),
            }
        )

        expected_array = cast(np.ndarray, expected)
        expected_df = pd.DataFrame(expected_array, index=contingency.index, columns=contingency.columns)
        residuals = (contingency - expected_df) / np.sqrt(expected_df)
        residual_path = RESIDUALS_DIR / f"chi_square_standardized_residuals_{column}.csv"
        residuals.to_csv(residual_path)
        logger.info("Saved chi-square residuals for %s to %s", column, residual_path)

    results_path = STATISTICS_DIR / "chi_square_results.csv"
    pd.DataFrame(tests).to_csv(results_path, index=False)
    logger.info("Saved chi-square results to %s", results_path)


def correlation_studies(df: pd.DataFrame, artifacts: PreprocessingArtifacts, logger) -> None:
    """Compute correlation matrices and export to disk."""
    logger.info("Calculating Spearman correlations")
    corr_numeric = df[artifacts.numeric_columns + ["AttritionFlag"]].corr(method="spearman")
    corr_path = STATISTICS_DIR / "spearman_correlations.csv"
    corr_numeric.to_csv(corr_path)
    logger.info("Saved full Spearman correlation matrix to %s", corr_path)

    attrition_corr = corr_numeric["AttritionFlag"].sort_values(ascending=False)
    attrition_corr_path = STATISTICS_DIR / "attrition_numeric_correlations.csv"
    attrition_corr.to_csv(attrition_corr_path)
    logger.info("Saved attrition correlations to %s", attrition_corr_path)


def build_eda_tables(df: pd.DataFrame, logger) -> None:
    """Create aggregated tables that are useful for analysis and dashboards."""
    logger.info("Building aggregated EDA tables")
    df = df.copy()
    df["AttritionRate"] = df["AttritionFlag"]

    attrition_by_jobrole = (
        df.groupby("JobRole")["AttritionRate"].mean().mul(100).sort_values(ascending=False).reset_index()
    )
    attrition_by_department = (
        df.groupby("Department")["AttritionRate"].mean().mul(100).sort_values(ascending=False).reset_index()
    )
    avg_income_by_education = (
        df.groupby(["EducationField", "Attrition"])["MonthlyIncome"].mean().round(2).reset_index()
    )
    distance_vs_jobrole = pd.crosstab(
        df["JobRole"],
        df["Attrition"],
        values=df["DistanceFromHome"],
        aggfunc="mean",
    )

    jobrole_path = TABLES_DIR / "attrition_rate_by_jobrole.csv"
    department_path = TABLES_DIR / "attrition_rate_by_department.csv"
    income_path = TABLES_DIR / "monthly_income_by_education_attrition.csv"
    distance_path = TABLES_DIR / "distance_from_home_by_jobrole_attrition.csv"

    attrition_by_jobrole.to_csv(jobrole_path, index=False)
    attrition_by_department.to_csv(department_path, index=False)
    avg_income_by_education.to_csv(income_path, index=False)
    distance_vs_jobrole.to_csv(distance_path)

    logger.info("Saved attrition by job role table to %s", jobrole_path)
    logger.info("Saved attrition by department table to %s", department_path)
    logger.info("Saved income by education table to %s", income_path)
    logger.info("Saved distance from home table to %s", distance_path)
