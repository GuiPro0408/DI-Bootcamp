from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, DIRECTORIES, METADATA_DIR


def prepare_output_structure(logger) -> None:
    """Create the nested output directory tree."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)


def load_data(
        path: Path = DATA_PATH, logger=None
) -> pd.DataFrame:
    """Load the dataset from disk and return a DataFrame."""
    if logger:
        logger.info("Loading dataset from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if logger:
        logger.info("Dataset loaded with shape %s", df.shape)
    return df


def write_metadata(
        metadata: dict, filename: str, logger=None
) -> None:
    """Write metadata dictionary to a JSON file in the metadata directory."""
    metadata_path = METADATA_DIR / filename
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if logger:
        logger.info("Wrote metadata to %s", metadata_path)
