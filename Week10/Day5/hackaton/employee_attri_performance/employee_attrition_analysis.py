"""Command-line entry point for the employee attrition analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - allow execution both as module and script
    from .pipeline import run_pipeline
except ImportError:  # pragma: no cover
    from pipeline import run_pipeline  # type: ignore


def main(dataset: str | None = None) -> None:
    """Run the full employee attrition analysis pipeline."""
    path = Path(dataset).expanduser().resolve() if dataset else None
    run_pipeline(path)


if __name__ == "__main__":
    main(*sys.argv[1:2])
