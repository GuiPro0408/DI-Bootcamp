"""Logging utilities for the generative AI pipeline."""

from __future__ import annotations

import logging
from typing import Optional

from config import LOG_DIR, LOGGER_NAME


def configure_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure console and file logging for the pipeline."""
    logger_name = name or LOGGER_NAME
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"
    logger = logging.getLogger(logger_name)
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
