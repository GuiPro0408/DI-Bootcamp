#!/usr/bin/env python3
"""Credit Card Fraud Detection Hackathon Pipeline.

This script automates the core steps required for the hackathon:
- Exploratory data analysis with Matplotlib, Seaborn, Plotly, and Plotnine.
- Feature engineering, preprocessing, and class-imbalance mitigation.
- Training and evaluation of multiple machine learning models.
- Export of curated artefacts ready for Tableau or PowerBI dashboards.

Example usage (from the project directory):
    python credit_fraud_pipeline.py --data-path creditcard.csv --use-smote

All generated files are placed inside the folder provided via --output-dir (defaults to ./outputs).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PLOTLY_AVAILABLE = False

try:
    from plotnine import (
        ggplot, aes, geom_histogram, labs, scale_x_continuous,
        theme_minimal, theme, guides, guide_legend,
    )

    PLOTNINE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PLOTNINE_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, balanced_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    IMBLEARN_AVAILABLE = False
    ImbPipeline = None  # type: ignore
    SMOTE = None  # type: ignore

sns.set_theme(style="whitegrid")
logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure root logger for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    if verbose:
        warnings.filterwarnings("ignore", category=UserWarning)


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    """Create a directory structure for outputs and return mapping."""
    subdirs = {
        "base": base_dir,
        "figures": base_dir / "figures",
        "reports": base_dir / "reports",
        "tables": base_dir / "tables",
        "dashboards": base_dir / "dashboards",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the credit card dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column identifying fraud labels.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features to boost model performance."""
    engineered = df.copy()
    if "Time" in engineered.columns:
        engineered["Hour"] = (engineered["Time"] / 3600.0) % 24.0
        engineered["DayFraction"] = (engineered["Time"] / (24 * 3600.0)) % 1.0
    if "Amount" in engineered.columns:
        engineered["Amount_Log"] = np.log1p(engineered["Amount"].clip(lower=0))
        engineered["Amount_Z"] = (engineered["Amount"] - engineered["Amount"].mean()) / (
                engineered["Amount"].std() + 1e-9
        )
    engineered["Transaction_Index"] = np.arange(len(engineered))
    return engineered


def save_dataframe_preview(df: pd.DataFrame, output_dir: Path, filename: str = "dataset_preview.csv") -> None:
    """Persist the first rows and descriptive statistics for quick reference."""
    preview_path = output_dir / filename
    desc_path = output_dir / "dataset_describe.csv"
    df.head(20).to_csv(preview_path, index=False)
    df.describe(include="all").transpose().to_csv(desc_path)
    logger.debug("Saved dataset preview to %s", preview_path)


def summarize_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a JSON summary with key dataset facts."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "class_balance": df["Class"].value_counts(normalize=True).to_dict(),
        "missing_values": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "transaction_amount_stats": df["Amount"].describe().to_dict() if "Amount" in df else {},
    }
    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info(
        "Rows: %s | Fraud proportion: %.4f",
        summary["n_rows"],
        summary["class_balance"].get(1, 0.0),
    )


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Class", data=df, palette=["#377eb8", "#e41a1c"], ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0])
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=50,
        log_scale=True,
        element="step",
        stat="density",
        common_norm=False,
        palette=["#377eb8", "#e41a1c"],
        ax=ax,
    )
    ax.set_title("Transaction Amount Distribution (Log Scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "amount_distribution.png", dpi=300)
    plt.close(fig)


def plot_hourly_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    if "Hour" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    hourly = df.groupby(["Hour", "Class"]).size().reset_index(name="TransactionCount")
    sns.lineplot(data=hourly, x="Hour", y="TransactionCount", hue="Class", ax=ax, marker="o")
    ax.set_title("Transactions per Hour by Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "hourly_transactions.png", dpi=300)
    plt.close(fig)


def plot_feature_correlations(df: pd.DataFrame, output_dir: Path, top_k: int = 12) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Class" not in numeric_cols:
        return
    corr_with_class = (
        df[numeric_cols]
        .corr()["Class"]
        .drop(labels=["Class"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    top_features = corr_with_class.head(top_k).index.tolist()
    if not top_features:
        return
    corr_matrix = df[top_features + ["Class"]].corr()
    fig, ax = plt.subplots(figsize=(0.8 * len(top_features) + 2, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Top Feature Correlations with Class")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlations.png", dpi=300)
    plt.close(fig)


def plot_amount_time_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    if "Hour" not in df.columns or "Amount_Log" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="Hour",
        y="Amount_Log",
        hue="Class",
        alpha=0.4,
        s=30,
        palette=["#377eb8", "#e41a1c"],
        ax=ax,
    )
    ax.set_title("Log Amount vs. Hour")
    ax.set_ylabel("Log1p Amount")
    fig.tight_layout()
    fig.savefig(output_dir / "hour_vs_amount_scatter.png", dpi=300)
    plt.close(fig)


def plot_interactive_visuals(df: pd.DataFrame, output_dir: Path) -> None:
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not installed; skipping interactive visualisations.")
        return
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="V2",
        y="V5",
        color=sample["Class"].map({0: "Legitimate", 1: "Fraud"}),
        title="Interactive View: V2 vs V5 by Class",
        opacity=0.4,
    )
    fig.write_html(str(output_dir / "plotly_v2_v5.html"))
    density_fig = px.density_heatmap(
        sample,
        x="Hour" if "Hour" in sample.columns else "Time",
        y="Amount",
        z="Class",
        nbinsx=24,
        nbinsy=30,
        title="Heatmap of Fraud Density",
        color_continuous_scale="Inferno",
    )
    density_fig.write_html(str(output_dir / "plotly_fraud_heatmap.html"))


def plot_plotnine_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    if not PLOTNINE_AVAILABLE:
        logger.warning("Plotnine is not installed; skipping Plotnine visualisation.")
        return
    sample = df.sample(n=min(20000, len(df)), random_state=42)
    plot = (
            ggplot(sample, aes("Amount", fill="factor(Class)"))
            + geom_histogram(alpha=0.6, bins=40, position="identity")
            + scale_x_continuous(trans="log10")
            + labs(
        title="Plotnine: Amount Distribution",
        x="Amount (log scale)",
        y="Count",
        fill="Class",
    )
            + theme_minimal()
            + theme(legend_position="top")
            + guides(fill=guide_legend(title="Class"))
    )
    plot.save(filename=str(output_dir / "plotnine_amount_distribution.png"), dpi=300)


def run_eda(df: pd.DataFrame, output_dirs: Dict[str, Path], sample_limit: int) -> None:
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs["figures"]
    save_dataframe_preview(df, tables_dir)
    summarize_dataset(df, output_dirs["reports"])
    sample = df.sample(n=min(sample_limit, len(df)), random_state=42)
    plot_class_distribution(sample, figures_dir)
    plot_amount_distribution(sample, figures_dir)
    plot_hourly_patterns(df, figures_dir)
    plot_feature_correlations(sample, figures_dir)
    plot_amount_time_scatter(sample, figures_dir)
    plot_interactive_visuals(sample, figures_dir)
    plot_plotnine_histogram(sample, figures_dir)


def build_models(
        random_state: int,
        use_smote: bool,
        smote_ratio: float,
        include_xgboost: bool,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if use_smote and IMBLEARN_AVAILABLE:
        logistic = ImbPipeline(
            steps=[
                ("sampler", SMOTE(random_state=random_state, sampling_strategy=smote_ratio)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        )
    else:
        if use_smote and not IMBLEARN_AVAILABLE:
            logger.warning("imblearn is not installed; falling back to class_weight balancing.")
        logistic = SkPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    models["logistic_regression"] = logistic

    models["random_forest"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    models["gradient_boosting"] = GradientBoostingClassifier(random_state=random_state)

    if include_xgboost and XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            tree_method="hist",
            eval_metric="auc",
        )
    elif include_xgboost:
        logger.warning("XGBoost is not installed; skipping XGBoost model.")
    return models


def get_prediction_scores(model: object, x: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            return scores
    return None


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_scores: Optional[np.ndarray]) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    return metrics


def extract_feature_importances(model: object, feature_names: List[str]) -> Optional[pd.DataFrame]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("classifier", estimator)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    if hasattr(estimator, "coef_"):
        coefs = np.ravel(estimator.coef_)
        return (
            pd.DataFrame({"feature": feature_names, "importance": np.abs(coefs)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return None


def compute_permutation_importance(
        model: object,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        random_state: int,
        sample_size: int = 5000,
) -> Optional[pd.DataFrame]:
    if sample_size < len(x_test):
        sample = x_test.sample(n=sample_size, random_state=random_state)
        target = y_test.loc[sample.index]
    else:
        sample = x_test
        target = y_test
    try:
        result = permutation_importance(
            model,
            sample,
            target,
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception as exc:  # pragma: no cover - diagnostic
        logger.warning("Permutation importance failed: %s", exc)
        return None
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Legitimate", "Fraud"], rotation=45)
    ax.set_yticklabels(["Legitimate", "Fraud"], rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(roc_entries: List[Tuple[np.ndarray, np.ndarray, float, str]], output_path: Path) -> None:
    if not roc_entries:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for fpr, tpr, auc_value, label in roc_entries:
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def train_and_evaluate(
        models: Dict[str, object],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        output_dirs: Dict[str, Path],
        random_state: int,
) -> Dict[str, Dict[str, float]]:
    metrics_summary: Dict[str, Dict[str, float]] = {}
    roc_entries: List[Tuple[np.ndarray, np.ndarray, float, str]] = []
    feature_tables = []
    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_scores = get_prediction_scores(model, x_test)
        metrics = compute_metrics(y_test, y_pred, y_scores)
        metrics_summary[name] = metrics

        report_text = classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
        report_path = output_dirs["reports"] / f"{name}_classification_report.txt"
        with report_path.open("w", encoding="utf-8") as fp:
            fp.write(report_text)

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, output_dirs["figures"] / f"{name}_confusion_matrix.png", name)

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc_value = roc_auc_score(y_test, y_scores)
            roc_entries.append((fpr, tpr, auc_value, name))

        fi_df = extract_feature_importances(model, feature_names)
        if fi_df is not None:
            fi_path = output_dirs["reports"] / f"{name}_feature_importances.csv"
            fi_df.to_csv(fi_path, index=False)
            feature_tables.append(fi_df.assign(model=name))

        if name == "random_forest":
            pi_df = compute_permutation_importance(
                model,
                x_test,
                y_test,
                feature_names,
                random_state=random_state,
                sample_size=min(8000, len(x_test)),
            )
            if pi_df is not None:
                pi_df.to_csv(output_dirs["reports"] / "random_forest_permutation_importance.csv", index=False)

    plot_roc_curves(roc_entries, output_dirs["figures"] / "roc_curves.png")

    if feature_tables:
        combined = pd.concat(feature_tables, ignore_index=True)
        combined.to_csv(output_dirs["reports"] / "feature_importances_combined.csv", index=False)

    metrics_path = output_dirs["reports"] / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_summary, fp, indent=2)
    metrics_df = pd.DataFrame(metrics_summary).transpose()
    metrics_df.to_csv(output_dirs["tables"] / "model_metrics.csv")

    return metrics_summary


def export_dashboard_assets(df: pd.DataFrame, output_dirs: Dict[str, Path]) -> None:
    dashboards_dir = output_dirs["dashboards"]
    class_summary = (
        df.groupby("Class")
        .agg(
            transaction_count=("Class", "size"),
            total_amount=("Amount", "sum"),
            average_amount=("Amount", "mean"),
        )
        .reset_index()
    )
    class_summary.to_csv(dashboards_dir / "class_summary.csv", index=False)

    if "Hour" in df.columns:
        hourly = (
            df.groupby(["Hour", "Class"])
            .agg(
                transaction_count=("Class", "size"),
                total_amount=("Amount", "sum"),
            )
            .reset_index()
        )
        hourly.to_csv(dashboards_dir / "hourly_trends.csv", index=False)

    if "Time" in df.columns:
        timeline = df.sort_values("Time").reset_index(drop=True)
    else:
        timeline = df.sort_index().reset_index(drop=True)
    timeline["Chrono"] = np.arange(len(timeline))
    timeline["Rolling_Fraud"] = timeline["Class"].rolling(window=500, min_periods=1).mean()
    timeline[["Chrono", "Rolling_Fraud"]].to_csv(dashboards_dir / "rolling_fraud_rate.csv", index=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end credit card fraud analysis pipeline.")
    parser.add_argument("--data-path", type=Path, default=Path("creditcard.csv"), help="Path to the CSV dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artefacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE oversampling (requires imblearn).")
    parser.add_argument(
        "--smote-ratio",
        type=float,
        default=0.5,
        help="Desired minority/majority ratio after SMOTE (0 < ratio <= 1).",
    )
    parser.add_argument(
        "--max-eda-sample",
        type=int,
        default=60000,
        help="Maximum number of rows sampled for EDA visualisations to keep plots readable.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Optional limit on total rows used for training/evaluation for quick iterations.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Train an XGBoost model (requires xgboost package).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis to speed up execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    configure_logging(args.verbose)
    sns.set_theme(style="whitegrid")

    if args.use_smote and not (0 < args.smote_ratio <= 1):
        raise ValueError("--smote-ratio must be between 0 and 1.")

    output_dirs = ensure_directories(args.output_dir)
    df_raw = load_data(args.data_path)
    df = engineer_features(df_raw)

    if not args.skip_eda:
        logger.info("Running exploratory data analysis...")
        run_eda(df, output_dirs, sample_limit=args.max_eda_sample)

    if args.train_sample_size is not None:
        if args.train_sample_size <= 0:
            raise ValueError("--train-sample-size must be positive.")
        if args.train_sample_size < len(df):
            logger.info("Subsampling dataset to %d rows for modelling and evaluation.", args.train_sample_size)
            df = df.sample(n=args.train_sample_size, random_state=args.random_state).reset_index(drop=True)
        else:
            logger.warning("Requested train sample size (%d) exceeds available rows (%d); using full dataset.",
                           args.train_sample_size, len(df))

    feature_cols = [col for col in df.columns if col != "Class"]
    x = df[feature_cols]
    y = df["Class"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    logger.info(
        "Training/test split complete -> train: %d rows, test: %d rows",
        len(x_train),
        len(x_test),
    )

    models = build_models(
        random_state=args.random_state,
        use_smote=args.use_smote,
        smote_ratio=args.smote_ratio,
        include_xgboost=args.include_xgboost,
    )

    metrics_summary = train_and_evaluate(
        models,
        x_train,
        y_train,
        x_test,
        y_test,
        feature_cols,
        output_dirs,
        random_state=args.random_state,
    )

    export_dashboard_assets(df, output_dirs)

    logger.info("Pipeline complete. Key metrics:")
    for model_name, metrics in metrics_summary.items():
        logger.info(
            "%s -> Recall: %.3f | Precision: %.3f | ROC-AUC: %.3f",
            model_name,
            metrics.get("recall", float("nan")),
            metrics.get("precision", float("nan")),
            metrics.get("roc_auc", float("nan")),
        )

    logger.info(
        "Next steps: Review reports in %s and connect the dashboards CSVs to Tableau or PowerBI.",
        output_dirs["base"],
    )


if __name__ == "__main__":
    main()
