"""Configuration constants for the document search and summarization system."""

from __future__ import annotations

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Output subdirectories
RAW_TEXTS_DIR = OUTPUT_DIR / "raw_texts"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
VECTOR_STORE_DIR = OUTPUT_DIR / "vector_store"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

# All directories to create
DIRECTORIES = [
    DATA_DIR,
    UPLOADS_DIR,
    OUTPUT_DIR,
    RAW_TEXTS_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    VECTOR_STORE_DIR,
    SUMMARIES_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOG_DIR,
]

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "facebook/bart-base"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Processing parameters
RANDOM_STATE = 42
CHUNK_SIZE = 400  # Target tokens per chunk
CHUNK_OVERLAP = 50  # Token overlap between chunks
BATCH_SIZE = 4  # CPU-optimized batch size for embeddings/summarization
TOP_K = 5  # Number of chunks to retrieve for search
MAX_DOCS = 50  # Maximum documents to process (CPU constraint)

# Summarization parameters
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_LENGTH = 150
SUMMARY_INPUT_MAX_TOKENS = 1024  # Max tokens to feed to summarizer

# Evaluation parameters
EVAL_NUM_QUERIES = 20  # Number of synthetic queries for evaluation
EVAL_TOP_K_VALUES = [1, 3, 5, 10]  # K values for precision@k and recall@k

# Logging
LOGGER_NAME = "document_search_pipeline"
