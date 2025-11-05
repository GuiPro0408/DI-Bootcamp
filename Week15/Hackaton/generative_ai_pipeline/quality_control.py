"""Quality control utilities for generated content."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import (
    QUALITY_MAX_LENGTH,
    QUALITY_MIN_LENGTH,
    QUALITY_REPORT_DIR,
    QUALITY_SIMILARITY_THRESHOLD,
    QUALITY_MODEL,
    EMBEDDING_MODEL,
    TEXT_MODEL,
)
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = AutoTokenizer = pipeline = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None  # type: ignore


@dataclass
class QualityResult:
    """Structured report holding quality control outcomes."""

    prompt: str
    text: str
    passed: bool
    similarity_score: Optional[float] = None
    perplexity: Optional[float] = None
    summary: Optional[str] = None
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "prompt": self.prompt,
            "text": self.text,
            "passed": self.passed,
            "similarity_score": self.similarity_score,
            "perplexity": self.perplexity,
            "summary": self.summary,
            "flags": self.flags,
        }


class QualityChecker:
    """Evaluate generated outputs against heuristic quality checks."""

    def __init__(
        self,
        summarizer_model: str = QUALITY_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        perplexity_model: str = TEXT_MODEL,
        similarity_threshold: float = QUALITY_SIMILARITY_THRESHOLD,
    ) -> None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for quality control. "
                "Install it via `pip install sentence-transformers`."
            )
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ImportError(
                "transformers and torch are required for quality control. "
                "Install them via `pip install transformers torch`."
            )

        self.summarizer_model = summarizer_model
        self.embedding_model_name = embedding_model
        self.perplexity_model_name = perplexity_model
        self.similarity_threshold = similarity_threshold

        self._summarizer = None
        self._embedder = None
        self._perplexity_model = None
        self._perplexity_tokenizer = None

        logger.info(
            "QualityChecker initialized",
            extra={
                "summarizer_model": summarizer_model,
                "embedding_model": embedding_model,
                "perplexity_model": perplexity_model,
            },
        )

    def load_summarizer(self):
        """Load summarization pipeline lazily."""
        if self._summarizer is None:
            try:
                logger.info("Loading summarization pipeline", extra={"model": self.summarizer_model})
                self._summarizer = pipeline("summarization", model=self.summarizer_model, device=-1)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning(
                    "Falling back to heuristic summarizer", extra={"model": self.summarizer_model, "error": str(exc)}
                )

                def fallback_summarizer(text: str, max_length: int = 60, **_: int):
                    tokens = text.split()
                    truncated = " ".join(tokens[: max_length // 2])
                    return [{"summary_text": truncated}]

                self._summarizer = fallback_summarizer
        return self._summarizer

    def load_embedder(self):
        """Load sentence embedding model lazily."""
        if self._embedder is None:
            logger.info("Loading sentence embedder", extra={"model": self.embedding_model_name})
            # Force CPU usage to avoid CUDA compatibility issues
            self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._embedder

    def load_perplexity_model(self):
        """Load causal language model for perplexity computation."""
        if self._perplexity_model is None or self._perplexity_tokenizer is None:
            logger.info("Loading perplexity model", extra={"model": self.perplexity_model_name})
            self._perplexity_tokenizer = AutoTokenizer.from_pretrained(self.perplexity_model_name)
            # Force CPU usage to avoid CUDA compatibility issues
            self._perplexity_model = AutoModelForCausalLM.from_pretrained(self.perplexity_model_name)
            if HAS_TORCH:
                self._perplexity_model.eval()
                # Explicitly move model to CPU
                self._perplexity_model = self._perplexity_model.cpu()
        return self._perplexity_model, self._perplexity_tokenizer

    def summarize(self, text: str, max_length: int = 60) -> Optional[str]:
        """Produce a concise summary using the configured summarizer."""
        if not text.strip():
            return None
        summarizer = self.load_summarizer()
        result = summarizer(text, max_length=max_length, min_length=15, do_sample=False)
        if isinstance(result, list):
            return result[0].get("summary_text")
        if isinstance(result, dict):
            return result.get("summary_text")
        return None

    def check_relevance(self, prompt: str, generated_text: str) -> Optional[float]:
        """Compute cosine similarity between prompt and generated text."""
        if not prompt.strip() or not generated_text.strip():
            return None

        embedder = self.load_embedder()
        vectors = embedder.encode([prompt, generated_text], convert_to_numpy=True)
        prompt_vec, generated_vec = vectors

        denom = np.linalg.norm(prompt_vec) * np.linalg.norm(generated_vec)
        if denom == 0:
            return None
        similarity = float(np.dot(prompt_vec, generated_vec) / denom)
        return similarity

    def compute_perplexity(self, text: str, model_name: Optional[str] = None) -> Optional[float]:
        """Compute perplexity for the provided text."""
        if not text.strip():
            return None
        model, tokenizer = self.load_perplexity_model()
        selected_model = model_name or self.perplexity_model_name

        inputs = tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].size(1) > 512:
            inputs = {k: v[:, :512] for k, v in inputs.items()}

        if HAS_TORCH:
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = float(torch.exp(loss).cpu().item())
        else:  # pragma: no cover - fallback path
            perplexity = math.inf

        logger.debug(
            "Computed perplexity",
            extra={"model": selected_model, "perplexity": perplexity},
        )
        return perplexity

    def validate_output(self, prompt: str, text: str) -> QualityResult:
        """Run heuristic checks and return a structured report."""
        flags: List[str] = []
        similarity = self.check_relevance(prompt, text)
        perplexity = self.compute_perplexity(text)
        summary = self.summarize(text)

        if similarity is not None and similarity < self.similarity_threshold:
            flags.append(f"low_similarity:{similarity:.2f}")
        if perplexity is not None and perplexity > 80:
            flags.append(f"high_perplexity:{perplexity:.1f}")
        if len(text.split()) < QUALITY_MIN_LENGTH:
            flags.append("too_short")
        if len(text.split()) > QUALITY_MAX_LENGTH:
            flags.append("too_long")

        passed = not flags
        result = QualityResult(
            prompt=prompt,
            text=text,
            passed=passed,
            similarity_score=similarity,
            perplexity=perplexity,
            summary=summary,
            flags=flags,
        )

        logger.info(
            "Quality validation complete",
            extra={
                "passed": passed,
                "similarity": similarity,
                "perplexity": perplexity,
                "flags": flags,
            },
        )
        return result

    def save_report(self, result: QualityResult) -> Path:
        """Persist a single quality report as JSON."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_quality.json"
        path = QUALITY_REPORT_DIR / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
        logger.debug("Saved quality report", extra={"path": str(path)})
        return path
