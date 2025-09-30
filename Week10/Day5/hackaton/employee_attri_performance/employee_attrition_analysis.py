"""Comprehensive employee attrition and performance analysis pipeline.

This script loads the IBM HR Analytics Employee Attrition dataset, performs
cleaning, preprocessing, exploratory data analysis (EDA), correlation studies,
statistical testing, and generates a collection of visual assets and summary
artifacts. It is designed to be run as a standalone tool for the hackathon
requirements."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:  # Optional dependency
    import seaborn as sns

    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", palette="deep")
except ImportError:  # pragma: no cover - optional dependency
    HAS_SEABORN = False
    plt.style.use("ggplot")

try:  # Optional dependency
    from scipy import stats

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SCIPY = False

try:  # Optional dependency
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PLOTLY = False

try:  # Optional dependency
    from plotnine import aes, geom_col, ggplot, labs, theme_minimal

    HAS_PLOTNINE = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PLOTNINE = False

DATA_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
STATISTICS_DIR = OUTPUT_DIR / "statistics"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"
REPORTS_DIR = OUTPUT_DIR / "reports"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOG_DIR = OUTPUT_DIR / "logs"
RESIDUALS_DIR = STATISTICS_DIR / "chi_square_residuals"

DIRECTORIES = [
    OUTPUT_DIR,
    SUMMARIES_DIR,
    PREPROCESSING_DIR,
    STATISTICS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    INTERACTIVE_DIR,
    REPORTS_DIR,
    METADATA_DIR,
    LOG_DIR,
    RESIDUALS_DIR,
]


@dataclass
class PreprocessingArtifacts:
    numeric_columns: list[str]
    categorical_columns: list[str]
    standardized_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    encoded_frame: pd.DataFrame


def configure_logging() -> logging.Logger:
    """Configure file and console logging for the pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"
    logger = logging.getLogger("employee_attrition_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def prepare_output_structure(logger: logging.Logger) -> None:
    """Create the nested output directory tree."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)


def load_data(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load the dataset from disk and return a DataFrame."""
    logger.info("Loading dataset from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    logger.info("Dataset loaded with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Basic data cleaning: drop duplicates and constant columns."""
    logger.info("Cleaning dataset: removing duplicates and constant columns")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        logger.info("Dropping constant columns: %s", constant_columns)
        df = df.drop(columns=constant_columns)

    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    metadata = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "constant_columns_removed": constant_columns,
    }
    metadata_path = METADATA_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote cleaning metadata to %s", metadata_path)

    return df


def summarize_dataset(df: pd.DataFrame, logger: logging.Logger) -> None:
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


def preprocess_features(df: pd.DataFrame, logger: logging.Logger) -> PreprocessingArtifacts:
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


def statistical_tests(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Run chi-square tests on selected categorical variables against attrition."""
    logger.info("Running chi-square tests on key categorical features")
    if not HAS_SCIPY:
        message = "SciPy not installed. Install scipy to run chi-square tests."
        (STATISTICS_DIR / "chi_square_results.txt").write_text(message, encoding="utf-8")
        logger.warning(message)
        return

    tests = []
    target = df["Attrition"]

    for column in ["OverTime", "JobRole", "Department", "BusinessTravel", "WorkLifeBalance"]:
        contingency = pd.crosstab(df[column], target)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        tests.append(
            {
                "feature": column,
                "chi2": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant_at_0.05": bool(p_value < 0.05),
            }
        )

        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
        residuals = (contingency - expected_df) / np.sqrt(expected_df)
        residual_path = RESIDUALS_DIR / f"chi_square_standardized_residuals_{column}.csv"
        residuals.to_csv(residual_path)
        logger.info("Saved chi-square residuals for %s to %s", column, residual_path)

    results_path = STATISTICS_DIR / "chi_square_results.csv"
    pd.DataFrame(tests).to_csv(results_path, index=False)
    logger.info("Saved chi-square results to %s", results_path)


def correlation_studies(df: pd.DataFrame, artifacts: PreprocessingArtifacts, logger: logging.Logger) -> None:
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


def build_eda_tables(df: pd.DataFrame, logger: logging.Logger) -> None:
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


def _bar_with_optional_seaborn(ax, data, x, y=None, hue=None, **kwargs) -> None:
    if HAS_SEABORN:
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    else:
        if hue:
            categories = sorted(data[hue].unique())
            positions = np.arange(data[x].nunique())
            width = 0.8 / max(len(categories), 1)
            colors = plt.cm.get_cmap("tab10", len(categories))
            for idx, category in enumerate(categories):
                subset = data[data[hue] == category]
                grouped = subset.groupby(x)[y].sum()
                offset_positions = positions[: len(grouped)] + (idx - (len(categories) - 1) / 2) * width
                ax.bar(offset_positions, grouped.values, width=width, color=colors(idx), label=str(category), alpha=0.8)
            ax.set_xticks(positions)
            ax.set_xticklabels(sorted(data[x].unique()), rotation=30, ha="right")
            ax.legend(title=hue)
        else:
            ax.bar(data[x], data[y], color="#4c72b0")


def create_visualizations(df: pd.DataFrame, artifacts: PreprocessingArtifacts, logger: logging.Logger) -> None:
    """Generate Matplotlib/Seaborn, Plotly, and Plotnine visualizations."""
    logger.info("Creating visualization assets")

    def save_fig(name: str, fig: plt.Figure) -> None:
        fig.tight_layout()
        fig_path = FIGURES_DIR / name
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure to %s", fig_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if HAS_SEABORN:
        sns.countplot(data=df, x="Attrition", ax=ax)
    else:
        counts = df["Attrition"].value_counts()
        ax.bar(counts.index, counts.values, color=["#4c72b0", "#dd8452"])
    ax.set_title("Employee Attrition Distribution")
    ax.set_ylabel("Number of Employees")
    save_fig("attrition_distribution.png", fig)

    attrition_by_dept = (
        df.groupby(["Department", "Attrition"])["AttritionFlag"].count().reset_index(name="Count")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    if HAS_SEABORN:
        sns.barplot(data=attrition_by_dept, x="Department", y="Count", hue="Attrition", ax=ax)
        ax.tick_params(axis="x", rotation=30)
    else:
        _bar_with_optional_seaborn(ax, attrition_by_dept, x="Department", y="Count", hue="Attrition")
    ax.set_title("Attrition by Department")
    ax.set_xlabel("")
    save_fig("attrition_by_department.png", fig)

    attrition_by_jobrole = (
        df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.barplot(x=attrition_by_jobrole.values, y=attrition_by_jobrole.index, ax=ax, palette="rocket")
    else:
        ax.barh(attrition_by_jobrole.index, attrition_by_jobrole.values, color="#c44d56")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("Job Role")
    ax.set_title("Attrition Rate by Job Role")
    save_fig("attrition_rate_by_jobrole.png", fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if HAS_SEABORN:
        sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax)
    else:
        data_groups = [df.loc[df["Attrition"] == label, "MonthlyIncome"].values for label in ["No", "Yes"]]
        ax.boxplot(data_groups, tick_labels=["No", "Yes"], patch_artist=True)
    ax.set_title("Monthly Income by Attrition Status")
    ax.set_ylabel("Monthly Income")
    save_fig("monthly_income_attrition_boxplot.png", fig)

    corr_matrix = df[artifacts.numeric_columns + ["AttritionFlag"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=ax, square=True)
    else:
        cax = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_yticklabels(corr_matrix.index)
    ax.set_title("Spearman Correlation Heatmap")
    save_fig("spearman_correlation_heatmap.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if HAS_SEABORN:
        sns.barplot(
            data=df,
            x="WorkLifeBalance",
            y="AttritionFlag",
            estimator=np.mean,
            errorbar="sd",
            ax=ax,
        )
    else:
        grouped = df.groupby("WorkLifeBalance")["AttritionFlag"].mean()
        ax.bar(grouped.index.astype(str), grouped.values, color="#4c72b0")
    ax.set_title("Attrition Rate by Work-Life Balance")
    ax.set_ylabel("Attrition Rate")
    save_fig("attrition_by_worklifebalance.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if HAS_SEABORN:
        sns.violinplot(data=df, x="Attrition", y="DistanceFromHome", inner="quartile", ax=ax)
    else:
        groups = [df.loc[df["Attrition"] == label, "DistanceFromHome"].values for label in ["No", "Yes"]]
        ax.violinplot(groups, showmeans=True, showextrema=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No", "Yes"])
    ax.set_title("Distance from Home vs Attrition")
    save_fig("distance_from_home_violin.png", fig)

    crosstab = pd.crosstab(df["JobRole"], df["Attrition"], normalize="index").mul(100)
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="viridis", ax=ax)
    else:
        cax = ax.imshow(crosstab, cmap="viridis", aspect="auto", vmin=0, vmax=100)
        for (i, j), value in np.ndenumerate(crosstab.to_numpy()):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white")
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(crosstab.columns)))
        ax.set_xticklabels(crosstab.columns)
        ax.set_yticks(range(len(crosstab.index)))
        ax.set_yticklabels(crosstab.index)
    ax.set_title("Attrition Percentage by Job Role")
    ax.set_xlabel("Attrition")
    save_fig("jobrole_attrition_heatmap.png", fig)

    if HAS_PLOTLY:
        plotly_fig = px.histogram(
            df,
            x="Age",
            color="Attrition",
            nbins=20,
            barmode="overlay",
            opacity=0.75,
            title="Attrition Distribution by Age",
        )
        plotly_fig.update_layout(template="plotly_white")
        plotly_path = INTERACTIVE_DIR / "plotly_attrition_age.html"
        plotly_fig.write_html(plotly_path, include_plotlyjs="cdn")
        logger.info("Saved Plotly histogram to %s", plotly_path)

    if HAS_PLOTNINE:
        attrition_department = (
            df.groupby("Department")["AttritionFlag"].mean().mul(100).reset_index()
        )
        plotnine_plot = (
                ggplot(attrition_department, aes("Department", "AttritionFlag"))
                + geom_col(fill="#17a2b8")
                + theme_minimal()
                + labs(
            title="Attrition Rate by Department (Plotnine)",
            x="Department",
            y="Attrition Rate (%)",
        )
        )
        plotnine_path = FIGURES_DIR / "plotnine_attrition_department.png"
        plotnine_plot.save(filename=str(plotnine_path), width=8, height=4, dpi=300)
        logger.info("Saved Plotnine visualization to %s", plotnine_path)


def generate_retention_recommendations(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Create a simple text report with actionable insights."""
    logger.info("Compiling retention recommendations")
    attrition_rate = df["AttritionFlag"].mean() * 100
    overtime = df.groupby("OverTime")["AttritionFlag"].mean().mul(100)
    job_role = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
    worklife = df.groupby("WorkLifeBalance")["AttritionFlag"].mean().mul(100)
    job_satisfaction = df.groupby("JobSatisfaction")["AttritionFlag"].mean().mul(100)

    top_roles = job_role.head(3)
    low_balance_levels = worklife.sort_values(ascending=False).head(2)

    lines = [
        f"Overall attrition rate: {attrition_rate:.2f}%",
        "",
        "Groups with elevated attrition:",
    ]

    lines.extend([f"- {role}: {rate:.2f}%" for role, rate in top_roles.items()])

    if "Yes" in overtime.index and "No" in overtime.index:
        gap = overtime["Yes"] - overtime["No"]
        if abs(gap) > 5:
            lines.append(f"- Overtime status gap: {gap:.2f} percentage points (Yes minus No)")

    lines.append("")
    lines.append("Work-life balance levels with higher attrition:")
    lines.extend([f"- Level {int(level)}: {rate:.2f}%" for level, rate in low_balance_levels.items()])

    average_satisfaction = job_satisfaction.mean()
    high_attrition_satisfaction = job_satisfaction[job_satisfaction > average_satisfaction]
    if not high_attrition_satisfaction.empty:
        lines.append("")
        lines.append("Job satisfaction levels exceeding the average attrition rate:")
        lines.extend(
            [
                f"- Level {int(level)}: {rate:.2f}%"
                for level, rate in high_attrition_satisfaction.sort_values(ascending=False).items()
            ]
        )

    lines.append("")
    lines.append("Recommended focus areas:")
    lines.append("1. Review overtime policies for high-risk roles and monitor workload peaks.")
    lines.append("2. Design targeted development plans and mentoring for roles above the company average.")
    lines.append("3. Expand flexible work options where work-life balance ≤ 2 and track improvements.")
    lines.append(
        "4. Launch quarterly pulse surveys aligned with job satisfaction drivers to capture qualitative feedback."
    )

    recommendations_path = REPORTS_DIR / "retention_recommendations.txt"
    recommendations_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved retention recommendations to %s", recommendations_path)


def main() -> None:
    logger = configure_logging()
    prepare_output_structure(logger)

    logger.info("Starting employee attrition analysis pipeline")
    df = load_data(DATA_PATH, logger)
    clean_df = clean_data(df, logger)
    summarize_dataset(clean_df, logger)
    artifacts = preprocess_features(clean_df, logger)
    statistical_tests(clean_df, logger)
    correlation_studies(clean_df, artifacts, logger)
    build_eda_tables(clean_df, logger)
    create_visualizations(clean_df, artifacts, logger)
    generate_retention_recommendations(clean_df, logger)

    logger.info("Analysis complete. Outputs available in: %s", OUTPUT_DIR)

    if not HAS_SEABORN:
        logger.warning("Seaborn not installed. Install seaborn for enhanced visual styling.")
    if not HAS_SCIPY:
        logger.warning("SciPy not installed. Chi-square tests were skipped.")
    if not HAS_PLOTLY:
        logger.warning("Plotly not installed. Install plotly to generate interactive HTML charts.")
    if not HAS_PLOTNINE:
        logger.warning("Plotnine not installed. Install plotnine to export ggplot-style charts.")


if __name__ == "__main__":
    main()
