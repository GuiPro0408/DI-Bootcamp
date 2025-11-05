"""Text generation utilities for the pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import (
    ALT_TEXT_MODEL,
    GENERATED_TEXT_DIR,
    TEXT_MODEL,
)
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    from transformers import pipeline  # type: ignore

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TRANSFORMERS = False
    pipeline = None  # type: ignore


@dataclass
class GenerationResult:
    """Container for generated text outputs."""

    prompt: str
    generated_text: str
    model_name: str
    temperature: float
    max_length: int
    generation_time: float

    def to_dict(self) -> Dict[str, object]:
        """Serialize result to JSON-friendly dict."""
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "generation_time": self.generation_time,
        }


class TextGenerator:
    """Wrapper around Hugging Face pipelines for text generation."""

    def __init__(
        self,
        default_model: str = TEXT_MODEL,
        alt_model: str = ALT_TEXT_MODEL,
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for text generation. "
                "Install it via `pip install transformers` and retry."
            )

        self._pipelines: Dict[str, object] = {}
        self.default_model = default_model
        self.alt_model = alt_model

        # Lazily instantiate pipelines to avoid heavy start-up costs
        self.logger = logger
        self.logger.info(
            "TextGenerator initialized",
            extra={"default_model": default_model, "alt_model": alt_model},
        )

    def _resolve_task(self, model_name: str) -> str:
        """Return the correct pipeline task for the given model."""
        if "t5" in model_name or "flan" in model_name:
            return "text2text-generation"
        return "text-generation"

    def load_model(self, model_name: Optional[str] = None) -> object:
        """Instantiate and cache the transformers pipeline."""
        model_key = model_name or self.default_model
        if model_key in self._pipelines:
            return self._pipelines[model_key]

        task = self._resolve_task(model_key)
        self.logger.info("Loading text generation pipeline", extra={"model": model_key, "task": task})
        generator = pipeline(task, model=model_key, device=-1)
        self._pipelines[model_key] = generator
        return generator

    def _generate(
        self,
        prompt: str,
        temperature: float,
        max_length: int,
        num_sequences: int,
        model_name: str,
    ) -> tuple[List[str], float]:
        """Internal helper to generate using the configured pipeline."""
        generator = self.load_model(model_name)
        task = self._resolve_task(model_name)
        
        # Use max_new_tokens instead of max_length to avoid conflicts
        params = {
            "temperature": temperature,
            "max_new_tokens": max_length,  # Changed from max_length
            "num_return_sequences": num_sequences,
            "do_sample": True,
            "truncation": True,  # Explicitly enable truncation
        }
        start_time = time.perf_counter()
        outputs = generator(prompt, **params)
        elapsed = time.perf_counter() - start_time

        texts: List[str] = []
        for item in outputs:
            if task == "text2text-generation":
                texts.append(item.get("generated_text", ""))
            else:
                texts.append(item.get("generated_text", ""))

        self.logger.debug(
            "Generated text",
            extra={
                "model": model_name,
                "prompt_preview": prompt[:80],
                "num_sequences": num_sequences,
                "elapsed": round(elapsed, 3),
            },
        )
        return texts, elapsed

    def generate_from_prompt(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_length: int = 100,
        num_sequences: int = 1,
        model_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Generate text from a single prompt."""
        target_model = model_name or self.default_model
        texts, elapsed = self._generate(prompt, temperature, max_length, num_sequences, target_model)
        return [
            GenerationResult(
                prompt=prompt,
                generated_text=text,
                model_name=target_model,
                temperature=temperature,
                max_length=max_length,
                generation_time=elapsed,
            )
            for text in texts
        ]

    def batch_generate(
        self,
        prompts: Iterable[str],
        batch_size: int = 4,
        temperature: float = 0.7,
        max_length: int = 100,
        model_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Generate text for a batch of prompts with progress logging."""
        target_model = model_name or self.default_model
        results: List[GenerationResult] = []
        prompt_list = list(prompts)
        total = len(prompt_list)

        for index in range(0, total, batch_size):
            batch = prompt_list[index : index + batch_size]
            self.logger.info(
                "Generating batch",
                extra={
                    "model": target_model,
                    "batch_start": index,
                    "batch_end": index + len(batch),
                    "total_prompts": total,
                },
            )
            for prompt in batch:
                results.extend(
                    self.generate_from_prompt(
                        prompt=prompt,
                        temperature=temperature,
                        max_length=max_length,
                        num_sequences=1,
                        model_name=target_model,
                    )
                )

        return results

    def save_results(self, results: List[GenerationResult], model_name: Optional[str] = None) -> Path:
        """Persist generation results to the outputs directory."""
        if not results:
            raise ValueError("No generation results to save.")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_model = model_name or results[0].model_name
        filename = f"{timestamp}_{target_model.replace('/', '_')}.json"
        path = GENERATED_TEXT_DIR / filename

        payload = [result.to_dict() for result in results]
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        self.logger.info(
            "Saved generation outputs",
            extra={"path": str(path), "num_records": len(results), "model": target_model},
        )
        return path
