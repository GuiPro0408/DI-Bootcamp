"""Shared IO utilities for the heart disease mini-project."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> None:
    """Create the directory at ``path`` if it doesn't exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """Persist ``obj`` to ``path`` in JSON format using UTF-8 encoding."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_csv(path: str) -> pd.DataFrame:
    """Thin wrapper around ``pandas.read_csv`` to keep IO concerns together."""
    return pd.read_csv(path)
