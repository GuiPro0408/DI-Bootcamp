from __future__ import annotations

import logging

try:  # pragma: no cover
    from .config import LOG_DIR, LOGGER_NAME
except ImportError:  # pragma: no cover
    from config import LOG_DIR, LOGGER_NAME


def configure_logging(name: str = LOGGER_NAME) -> logging.Logger:
    """Configure file and console logging for the pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"
    logger = logging.getLogger(name)
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
