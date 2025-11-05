"""Data loading and persistence utilities for the generative AI pipeline."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from config import OUTPUT_DIR, RANDOM_STATE
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    from datasets import load_dataset  # type: ignore

    HAS_DATASETS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_DATASETS = False
    load_dataset = None  # type: ignore


@dataclass
class DatasetSplit:
    """Container for dataset prompts and references."""

    prompts: List[str]
    references: List[str]


def _stratified_indices(labels: Sequence[int], sample_size: int, seed: int) -> List[int]:
    """Return stratified sample indices for binary IMDB labels."""
    grouped: dict[int, List[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(idx)

    rng = random.Random(seed)
    selected: List[int] = []
    per_class = max(1, sample_size // max(1, len(grouped)))

    for label, indices in grouped.items():
        rng.shuffle(indices)
        selected.extend(indices[:per_class])

    if len(selected) > sample_size:
        rng.shuffle(selected)
        selected = selected[:sample_size]
    elif len(selected) < sample_size:
        remaining = sample_size - len(selected)
        reservoir = [idx for indices in grouped.values() for idx in indices]
        rng.shuffle(reservoir)
        selected.extend(reservoir[:remaining])

    return sorted(set(selected))


def load_imdb_subset(
    split: str = "train",
    sample_pct: float = 0.05,
    stratify: bool = True,
    seed: int = RANDOM_STATE,
) -> DatasetSplit:
    """Load a (optionally stratified) subset of the IMDB dataset."""
    if not HAS_DATASETS:
        raise ImportError(
            "The `datasets` library is required to load the IMDB dataset. "
            "Install it via `pip install datasets` and retry."
        )

    dataset = load_dataset("imdb", split=split)
    total_samples = len(dataset)
    sample_size = max(1, int(total_samples * sample_pct))

    if stratify and "label" in dataset.column_names:
        labels = [int(example["label"]) for example in dataset]
        selected_indices = _stratified_indices(labels, sample_size, seed)
    else:
        rng = random.Random(seed)
        selected_indices = rng.sample(range(total_samples), sample_size)

    prompts: List[str] = []
    references: List[str] = []
    for idx in selected_indices:
        example = dataset[int(idx)]
        text = example["text"]
        prompts.append(text)
        references.append(text)

    logger.info(
        "Loaded IMDB subset",
        extra={
            "split": split,
            "sample_pct": sample_pct,
            "sample_size": sample_size,
            "stratified": stratify,
        },
    )

    return DatasetSplit(prompts=prompts, references=references)


def save_json(data: Iterable, path: Path) -> None:
    """Persist JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    logger.debug("Saved JSON artifact", extra={"path": str(path)})


def load_json(path: Path) -> Iterable:
    """Load JSON data from disk."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    logger.debug("Loaded JSON artifact", extra={"path": str(path)})
    return payload


def save_prompts(prompts: Sequence[str], path: Path | None = None) -> Path:
    """Convenience helper to persist prompts JSON."""
    target_path = path or (OUTPUT_DIR / "prompts.json")
    save_json({"prompts": list(prompts)}, target_path)
    return target_path


def save_references(references: Sequence[str], path: Path | None = None) -> Path:
    """Convenience helper to persist reference texts JSON."""
    target_path = path or (OUTPUT_DIR / "references.json")
    save_json({"references": list(references)}, target_path)
    return target_path
