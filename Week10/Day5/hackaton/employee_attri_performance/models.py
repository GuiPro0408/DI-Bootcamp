from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class PreprocessingArtifacts:
    numeric_columns: list[str]
    categorical_columns: list[str]
    standardized_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    encoded_frame: pd.DataFrame
