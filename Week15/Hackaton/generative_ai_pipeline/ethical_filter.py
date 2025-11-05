"""Ethical filtering for generated content."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import ETHICAL_FLAGS_DIR, ETHICAL_SEVERITY_THRESHOLD, TOXICITY_MODEL
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    from transformers import pipeline  # type: ignore

    HAS_TOXIC_BERT = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TOXIC_BERT = False
    pipeline = None  # type: ignore

# Basic keyword patterns; intentionally conservative and extendable.
PROFANITY_PATTERNS = [
    r"\b(?:damn|hell|shit|fuck)\b",
    r"\b(?:bitch|bastard|asshole)\b",
]

HATE_SPEECH_PATTERNS = [
    r"\b(?:kill|exterminate)\s+(?:them|immigrants|gays|muslims)\b",
    r"\b(?:white power|great replacement|blood and soil)\b",
]

BIAS_PATTERNS = [
    r"\bfemale\s+nurse\b",
    r"\bmale\s+engineer\b",
    r"\b(?:women|men)\s+are\s+(?:better|worse)\b",
]


@dataclass
class FilterResult:
    """Structured outcome for ethical filtering."""

    text: str
    is_flagged: bool
    severity: float
    categories: List[str] = field(default_factory=list)
    matched_patterns: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        return {
            "text": self.text,
            "is_flagged": self.is_flagged,
            "severity": self.severity,
            "categories": self.categories,
            "matched_patterns": self.matched_patterns,
        }


class EthicalFilter:
    """Combine rule-based and model-based filtering for generated text."""

    def __init__(
        self,
        severity_threshold: float = ETHICAL_SEVERITY_THRESHOLD,
        toxicity_model: str = TOXICITY_MODEL,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.toxicity_model = toxicity_model

        self._classifier = None
        self._compiled_patterns = {
            "profanity": [re.compile(pattern, flags=re.IGNORECASE) for pattern in PROFANITY_PATTERNS],
            "hate_speech": [re.compile(pattern, flags=re.IGNORECASE) for pattern in HATE_SPEECH_PATTERNS],
            "bias": [re.compile(pattern, flags=re.IGNORECASE) for pattern in BIAS_PATTERNS],
        }

        if HAS_TOXIC_BERT:
            logger.info("EthicalFilter initialized with classifier", extra={"model": toxicity_model})
        else:
            logger.warning("EthicalFilter running without toxicity classifier; only rule-based checks available.")

    def load_classifier(self):
        """Load optional toxicity classifier pipeline."""
        if not HAS_TOXIC_BERT:
            return None
        if self._classifier is None:
            logger.info("Loading toxicity classifier", extra={"model": self.toxicity_model})
            # Force CPU usage to avoid CUDA compatibility issues
            self._classifier = pipeline("text-classification", model=self.toxicity_model, device=-1)
        return self._classifier

    def _rule_based_scan(self, text: str) -> Dict[str, List[str]]:
        """Run regex-based checks against the text."""
        matches: Dict[str, List[str]] = {}
        for category, patterns in self._compiled_patterns.items():
            category_matches: List[str] = []
            for pattern in patterns:
                for match in pattern.findall(text):
                    category_matches.append(match if isinstance(match, str) else match[0])
            if category_matches:
                matches[category] = category_matches
        return matches

    def _classifier_scan(self, text: str) -> Optional[float]:
        """Compute toxicity score via classifier if available."""
        classifier = self.load_classifier()
        if not classifier:
            return None
        predictions = classifier(text, truncation=True)
        if isinstance(predictions, list) and predictions:
            score = float(predictions[0].get("score", 0.0))
            label = predictions[0].get("label", "")
            if "toxic" not in label.lower():
                # Some classifiers output labels like TOXIC/NON_TOXIC. Treat non-toxic as low severity.
                score = 1.0 - score
            return score
        return None

    def scan_text(self, text: str) -> FilterResult:
        """Run ethical filtering and return normalized result."""
        matched_patterns = self._rule_based_scan(text)
        categories = list(matched_patterns.keys())

        classifier_score = self._classifier_scan(text)
        severity = max(
            [classifier_score or 0.0]
            + [0.7 if cat == "hate_speech" else 0.5 for cat in categories]
        )

        is_flagged = severity >= self.severity_threshold or bool(categories)
        if classifier_score is not None and classifier_score >= self.severity_threshold:
            if "toxicity" not in categories:
                categories.append("toxicity")

        result = FilterResult(
            text=text,
            is_flagged=is_flagged,
            severity=float(severity),
            categories=categories,
            matched_patterns=matched_patterns,
        )

        logger.info(
            "Ethical scan complete",
            extra={
                "flagged": result.is_flagged,
                "severity": result.severity,
                "categories": result.categories,
            },
        )
        return result

    def save_flagged(self, result: FilterResult, metadata: Optional[dict] = None) -> Path:
        """Persist flagged results to disk."""
        if not result.is_flagged:
            raise ValueError("Only flagged results should be persisted via save_flagged.")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_flagged.json"
        path = ETHICAL_FLAGS_DIR / filename

        payload = {
            "result": result.to_dict(),
            "metadata": metadata or {},
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        logger.debug("Persisted ethical flag", extra={"path": str(path)})
        return path
