from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from typing import Any, Callable, Sequence, cast

try:  # pragma: no cover
    from .config import FIGURES_DIR, INTERACTIVE_DIR
    from .models import PreprocessingArtifacts
except ImportError:  # pragma: no cover
    from config import FIGURES_DIR, INTERACTIVE_DIR
    from models import PreprocessingArtifacts

try:  # Optional dependency
    import seaborn as sns

    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", palette="deep")
except ImportError:  # pragma: no cover - optional dependency
    HAS_SEABORN = False
    sns = None
    plt.style.use("ggplot")

try:  # Optional dependency
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PLOTLY = False
    px = None

try:  # Optional dependency
    from plotnine import aes, geom_col, ggplot, labs, theme_minimal

    HAS_PLOTNINE = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PLOTNINE = False
    aes = geom_col = ggplot = labs = theme_minimal = None


def _bar_with_optional_seaborn(ax, data, x, y=None, hue=None, **kwargs) -> None:
    if HAS_SEABORN and sns is not None:
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


def create_visualizations(df: pd.DataFrame, artifacts: PreprocessingArtifacts, logger) -> None:
    """Generate Matplotlib/Seaborn, Plotly, and Plotnine visualizations."""
    logger.info("Creating visualization assets")

    def save_fig(name: str, fig: Figure) -> None:
        fig.tight_layout()
        fig_path = FIGURES_DIR / name
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure to %s", fig_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if HAS_SEABORN and sns is not None:
        sns.countplot(data=df, x="Attrition", ax=ax)
    else:
        counts = df["Attrition"].value_counts()
        ax.bar(list(counts.index), counts.to_numpy(), color=["#4c72b0", "#dd8452"])
    ax.set_title("Employee Attrition Distribution")
    ax.set_ylabel("Number of Employees")
    save_fig("attrition_distribution.png", fig)

    attrition_by_dept = (
        df.groupby(["Department", "Attrition"])["AttritionFlag"].count().reset_index(name="Count")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    if HAS_SEABORN and sns is not None:
        sns.barplot(data=attrition_by_dept, x="Department", y="Count", hue="Attrition", ax=ax)
        ax.tick_params(axis="x", rotation=30)
    else:
        _bar_with_optional_seaborn(ax, attrition_by_dept, x="Department", y="Count", hue="Attrition")
    ax.set_title("Attrition by Department")
    ax.set_xlabel("")
    save_fig("attrition_by_department.png", fig)

    attrition_by_jobrole = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN and sns is not None:
        sns.barplot(x=attrition_by_jobrole.values, y=attrition_by_jobrole.index, ax=ax, palette="rocket")
    else:
        ax.barh(list(attrition_by_jobrole.index), attrition_by_jobrole.to_numpy(), color="#c44d56")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("Job Role")
    ax.set_title("Attrition Rate by Job Role")
    save_fig("attrition_rate_by_jobrole.png", fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if HAS_SEABORN and sns is not None:
        sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax)
    else:
        data_groups: Sequence[np.ndarray] = [
            np.asarray(df.loc[df["Attrition"] == label, "MonthlyIncome"].to_numpy())
            for label in ["No", "Yes"]
        ]
        ax.boxplot(data_groups, tick_labels=["No", "Yes"], patch_artist=True)
    ax.set_title("Monthly Income by Attrition Status")
    ax.set_ylabel("Monthly Income")
    save_fig("monthly_income_attrition_boxplot.png", fig)

    corr_matrix = df[artifacts.numeric_columns + ["AttritionFlag"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN and sns is not None:
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
    if HAS_SEABORN and sns is not None:
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
        ax.bar(grouped.index.astype(str).tolist(), grouped.to_numpy(), color="#4c72b0")
    ax.set_title("Attrition Rate by Work-Life Balance")
    ax.set_ylabel("Attrition Rate")
    save_fig("attrition_by_worklifebalance.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if HAS_SEABORN and sns is not None:
        sns.violinplot(data=df, x="Attrition", y="DistanceFromHome", inner="quartile", ax=ax)
    else:
        groups: Sequence[np.ndarray] = [
            np.asarray(df.loc[df["Attrition"] == label, "DistanceFromHome"].to_numpy())
            for label in ["No", "Yes"]
        ]
        ax.violinplot(groups, showmeans=True, showextrema=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No", "Yes"])
    ax.set_title("Distance from Home vs Attrition")
    save_fig("distance_from_home_violin.png", fig)

    crosstab = pd.crosstab(df["JobRole"], df["Attrition"], normalize="index").mul(100)
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN and sns is not None:
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

    if HAS_PLOTLY and px is not None:
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

    if HAS_PLOTNINE and all(obj is not None for obj in (aes, geom_col, ggplot, labs, theme_minimal)):
        attrition_department = (
            df.groupby("Department")["AttritionFlag"].mean().mul(100).reset_index()
        )
        plotnine_ggplot = cast(Callable[..., Any], ggplot)
        plotnine_aes = cast(Callable[..., Any], aes)
        plotnine_geom_col = cast(Callable[..., Any], geom_col)
        plotnine_theme_minimal = cast(Callable[..., Any], theme_minimal)
        plotnine_labs = cast(Callable[..., Any], labs)

        plotnine_plot = (
            plotnine_ggplot(attrition_department, plotnine_aes("Department", "AttritionFlag"))
            + plotnine_geom_col(fill="#17a2b8")
            + plotnine_theme_minimal()
            + plotnine_labs(
                title="Attrition Rate by Department (Plotnine)",
                x="Department",
                y="Attrition Rate (%)",
            )
        )
        plotnine_path = FIGURES_DIR / "plotnine_attrition_department.png"
        plotnine_plot.save(filename=str(plotnine_path), width=8, height=4, dpi=300)
        logger.info("Saved Plotnine visualization to %s", plotnine_path)
