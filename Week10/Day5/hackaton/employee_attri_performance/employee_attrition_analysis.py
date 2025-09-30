"""Comprehensive employee attrition and performance analysis pipeline.

This script loads the IBM HR Analytics Employee Attrition dataset, performs
cleaning, preprocessing, exploratory data analysis (EDA), correlation studies,
statistical testing, and generates a collection of visual assets and summary
artifacts. It is designed to be run as a standalone tool for the hackathon
requirements.
"""

from __future__ import annotations

import json
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


@dataclass
class PreprocessingArtifacts:
    numeric_columns: list[str]
    categorical_columns: list[str]
    standardized_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    encoded_frame: pd.DataFrame


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from disk and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: drop duplicates and constant columns."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)

    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    metadata = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "constant_columns_removed": constant_columns,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """Persist high-level dataset summaries to the outputs folder."""
    numeric_summary = df.describe().transpose()
    categorical_summary = df.select_dtypes(include="object").describe().transpose()
    attrition_counts = df["Attrition"].value_counts().rename_axis("Attrition").to_frame("Count")
    attrition_rate = df["Attrition"].value_counts(normalize=True).mul(100).round(2)

    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")
    categorical_summary.to_csv(OUTPUT_DIR / "categorical_summary.csv")
    attrition_counts.to_csv(OUTPUT_DIR / "attrition_distribution.csv")

    with (OUTPUT_DIR / "quick_facts.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            dedent(
                f"""
                Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
                Overall attrition rate: {attrition_rate.get('Yes', 0):.2f}%
                """
            ).strip()
        )


def preprocess_features(df: pd.DataFrame) -> PreprocessingArtifacts:
    """Generate standardized, normalized, and encoded feature sets."""
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != "AttritionFlag"]
    categorical_columns = [col for col in df.select_dtypes(include="object").columns if col != "Attrition"]

    numeric_frame = df[numeric_columns].copy()
    standardized = (numeric_frame - numeric_frame.mean()) / numeric_frame.std(ddof=0)
    standardized = standardized.add_suffix("_zscore")

    normalized = (numeric_frame - numeric_frame.min()) / (numeric_frame.max() - numeric_frame.min())
    normalized = normalized.add_suffix("_minmax")

    encoded = pd.get_dummies(df[categorical_columns], drop_first=True)

    standardized.to_csv(OUTPUT_DIR / "numeric_standardized.csv", index=False)
    normalized.to_csv(OUTPUT_DIR / "numeric_normalized.csv", index=False)
    encoded.to_csv(OUTPUT_DIR / "categorical_encoded.csv", index=False)

    model_ready = pd.concat([standardized, encoded, df[["AttritionFlag"]].reset_index(drop=True)], axis=1)
    model_ready.to_csv(OUTPUT_DIR / "model_ready_features.csv", index=False)

    return PreprocessingArtifacts(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        standardized_frame=standardized,
        normalized_frame=normalized,
        encoded_frame=encoded,
    )


def statistical_tests(df: pd.DataFrame) -> None:
    """Run chi-square tests on selected categorical variables against attrition."""
    if not HAS_SCIPY:
        (OUTPUT_DIR / "chi_square_results.txt").write_text(
            "SciPy not installed. Install scipy to run chi-square tests.",
            encoding="utf-8",
        )
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
        residuals.to_csv(OUTPUT_DIR / f"chi_square_standardized_residuals_{column}.csv")

    pd.DataFrame(tests).to_csv(OUTPUT_DIR / "chi_square_results.csv", index=False)


def correlation_studies(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> None:
    """Compute correlation matrices and export to disk."""
    corr_numeric = df[artifacts.numeric_columns + ["AttritionFlag"]].corr(method="spearman")
    corr_numeric.to_csv(OUTPUT_DIR / "spearman_correlations.csv")

    attrition_corr = corr_numeric["AttritionFlag"].sort_values(ascending=False)
    attrition_corr.to_csv(OUTPUT_DIR / "attrition_numeric_correlations.csv")


def build_eda_tables(df: pd.DataFrame) -> None:
    """Create aggregated tables that are useful for analysis and dashboards."""
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

    attrition_by_jobrole.to_csv(OUTPUT_DIR / "attrition_rate_by_jobrole.csv", index=False)
    attrition_by_department.to_csv(OUTPUT_DIR / "attrition_rate_by_department.csv", index=False)
    avg_income_by_education.to_csv(OUTPUT_DIR / "monthly_income_by_education_attrition.csv", index=False)
    distance_vs_jobrole.to_csv(OUTPUT_DIR / "distance_from_home_by_jobrole_attrition.csv")


def _bar_with_optional_seaborn(ax, data, x, y=None, hue=None, **kwargs) -> None:
    if HAS_SEABORN:
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    else:
        if hue:
            categories = sorted(data[hue].unique())
            base_positions = np.arange(data[x].nunique())
            offsets = np.linspace(-0.2, 0.2, len(categories))
            colors = plt.cm.get_cmap("tab10", len(categories))
            for idx, category in enumerate(categories):
                mask = data[hue] == category
                grouped = data[mask].groupby(x)[y].sum()
                positions = np.arange(len(grouped)) + offsets[idx]
                ax.bar(positions, grouped.values, width=0.4 / len(categories), color=colors(idx), label=category)
                ax.set_xticks(np.arange(len(grouped)))
                ax.set_xticklabels(grouped.index, rotation=30, ha="right")
            ax.legend(title=hue)
        else:
            ax.bar(data[x], data[y], color="#4c72b0")


def create_visualizations(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> None:
    """Generate Matplotlib/Seaborn, Plotly, and Plotnine visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_fig(name: str, fig: plt.Figure) -> None:
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
        plt.close(fig)

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
        for attr_value, subset in attrition_by_dept.groupby("Attrition"):
            ax.bar(subset["Department"], subset["Count"], label=attr_value, alpha=0.7)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Attrition")
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
        ax.boxplot(data_groups, labels=["No", "Yes"], patch_artist=True)
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
        plotly_fig.write_html(OUTPUT_DIR / "plotly_attrition_age.html", include_plotlyjs="cdn")

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
        plotnine_plot.save(filename=str(OUTPUT_DIR / "plotnine_attrition_department.png"), width=8, height=4, dpi=300)


def generate_retention_recommendations(df: pd.DataFrame) -> None:
    """Create a simple text report with actionable insights."""
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

    (OUTPUT_DIR / "retention_recommendations.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    clean_df = clean_data(df)
    summarize_dataset(clean_df)
    artifacts = preprocess_features(clean_df)
    statistical_tests(clean_df)
    correlation_studies(clean_df, artifacts)
    build_eda_tables(clean_df)
    create_visualizations(clean_df, artifacts)
    generate_retention_recommendations(clean_df)

    print(f"Analysis complete. Outputs available in: {OUTPUT_DIR}")
    if not HAS_SEABORN:
        print("Seaborn not installed. Install seaborn for enhanced visual styling.")
    if not HAS_SCIPY:
        print("SciPy not installed. Chi-square tests were skipped.")
    if not HAS_PLOTLY:
        print("Plotly not installed. Install plotly to generate interactive HTML charts.")
    if not HAS_PLOTNINE:
        print("Plotnine not installed. Install plotnine to export ggplot-style charts.")


if __name__ == "__main__":
    main()
