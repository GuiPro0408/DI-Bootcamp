from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
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

LOGGER_NAME = "employee_attrition_pipeline"
