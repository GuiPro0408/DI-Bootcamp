"""Configuration constants for the generative AI pipeline."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TEXT_MODEL = "distilgpt2"
ALT_TEXT_MODEL = "t5-small"
QUALITY_MODEL = "distilbert-base-uncased"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOXICITY_MODEL = "unitary/toxic-bert"

RANDOM_STATE = 42
BATCH_SIZE = 4

QUALITY_SIMILARITY_THRESHOLD = 0.5
QUALITY_MIN_LENGTH = 20
QUALITY_MAX_LENGTH = 512

ETHICAL_SEVERITY_THRESHOLD = 0.6

LOG_DIR = OUTPUT_DIR / "logs"
GENERATED_TEXT_DIR = OUTPUT_DIR / "generated_texts"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
QUALITY_REPORT_DIR = OUTPUT_DIR / "quality_reports"
ETHICAL_FLAGS_DIR = OUTPUT_DIR / "ethical_flags"
EVALUATION_DIR = OUTPUT_DIR / "evaluation_reports"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
MODEL_DIR = OUTPUT_DIR / "models"

LOGGER_NAME = "generative_ai_pipeline"

for directory in (
    OUTPUT_DIR,
    LOG_DIR,
    GENERATED_TEXT_DIR,
    SUMMARY_DIR,
    QUALITY_REPORT_DIR,
    ETHICAL_FLAGS_DIR,
    EVALUATION_DIR,
    IMAGE_OUTPUT_DIR,
    MODEL_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
