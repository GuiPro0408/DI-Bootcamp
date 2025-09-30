from __future__ import annotations

import warnings
from pathlib import Path

from analysis import HAS_SCIPY, build_eda_tables, correlation_studies, statistical_tests
from config import DATA_PATH, OUTPUT_DIR
from data_processing import clean_data, preprocess_features, summarize_dataset
from io_utils import load_data, prepare_output_structure
from logging_utils import configure_logging
from reporting import generate_retention_recommendations
from visualization import HAS_PLOTLY, HAS_PLOTNINE, HAS_SEABORN, create_visualizations

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_pipeline(dataset_path: Path | None = None) -> None:
    """Execute the full employee attrition analysis pipeline."""
    logger = configure_logging()
    prepare_output_structure(logger)

    target_path = dataset_path or DATA_PATH
    logger.info("Starting employee attrition analysis pipeline")
    df = load_data(target_path, logger=logger)
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
