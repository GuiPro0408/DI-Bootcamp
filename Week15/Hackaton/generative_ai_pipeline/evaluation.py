"""Evaluation helpers for generated content."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from config import EVALUATION_DIR
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    HAS_NLTK = True
except Exception:  # pragma: no cover - optional dependency
    HAS_NLTK = False
    SmoothingFunction = None  # type: ignore
    sentence_bleu = None  # type: ignore

try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_ROUGE = False
    rouge_scorer = None  # type: ignore


def compute_bleu(generated: Sequence[str], references: Sequence[str]) -> Optional[float]:
    """Compute average BLEU score between generated and reference texts."""
    if not HAS_NLTK:
        logger.warning("NLTK is not available; BLEU computation skipped.")
        return None
    if not generated or len(generated) != len(references):
        return None

    smoothie = SmoothingFunction().method1
    scores: List[float] = []
    for hyp, ref in zip(generated, references):
        ref_tokens = [ref.split()]
        hyp_tokens = hyp.split()
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        scores.append(score)

    bleu = float(sum(scores) / len(scores))
    logger.debug("Computed BLEU score", extra={"bleu": bleu})
    return bleu


def compute_rouge(generated: Sequence[str], references: Sequence[str]) -> Optional[Dict[str, float]]:
    """Compute ROUGE-1/2/L scores for generated text."""
    if not HAS_ROUGE:
        logger.warning("rouge-score is not available; ROUGE computation skipped.")
        return None
    if not generated or len(generated) != len(references):
        return None

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregate = defaultdict(float)

    for hyp, ref in zip(generated, references):
        scores = scorer.score(ref, hyp)
        for metric, value in scores.items():
            aggregate[metric] += value.fmeasure

    count = len(generated)
    rouge = {metric: value / count for metric, value in aggregate.items()}
    logger.debug("Computed ROUGE scores", extra=rouge)
    return rouge


def compute_perplexity(texts: Sequence[str], model_tokenizer: Optional[tuple]) -> Optional[float]:
    """Compute average perplexity given a (model, tokenizer) tuple."""
    if model_tokenizer is None:
        return None
    model, tokenizer = model_tokenizer

    try:
        import torch  # Imported lazily to avoid mandatory dependency
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("torch is not available; perplexity computation skipped.")
        return None

    perplexities: List[float] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].size(1) > 512:
            inputs = {k: v[:, :512] for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexities.append(float(torch.exp(outputs.loss).cpu().item()))

    avg_perplexity = float(sum(perplexities) / len(perplexities))
    logger.debug("Computed evaluation perplexity", extra={"perplexity": avg_perplexity})
    return avg_perplexity


def generate_adversarial_prompts(base_prompts: Sequence[str], noise_level: float = 0.2) -> List[str]:
    """Create noisy variants of prompts to test robustness."""
    rng = random.Random(42)
    adversarial_prompts: List[str] = []

    for prompt in base_prompts:
        tokens = prompt.split()
        for idx in range(len(tokens)):
            if rng.random() < noise_level:
                # Apply simple perturbations: swap, insert typo, or add bias trigger
                choice = rng.choice(["swap", "typo", "bias"])
                if choice == "swap" and idx < len(tokens) - 1:
                    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
                elif choice == "typo":
                    tokens[idx] = tokens[idx] + rng.choice(["...", "??", "!"])
                else:
                    tokens.insert(idx, rng.choice(["female", "male", "rich", "poor"]))
        adversarial_prompts.append(" ".join(tokens))

    logger.info("Generated adversarial prompts", extra={"count": len(adversarial_prompts)})
    return adversarial_prompts


def robustness_test(
    generator_fn: Callable[[str], Sequence[str]],
    adversarial_prompts: Sequence[str],
) -> Dict[str, float]:
    """Measure how average length changes under adversarial prompts."""
    lengths: List[int] = []
    for prompt in adversarial_prompts:
        outputs = generator_fn(prompt)
        if outputs:
            lengths.append(len(outputs[0]))
    avg_length = float(sum(lengths) / len(lengths)) if lengths else 0.0
    logger.info("Robustness test complete", extra={"avg_length": avg_length, "num_prompts": len(adversarial_prompts)})
    return {"avg_length": avg_length, "num_prompts": len(adversarial_prompts)}


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics."""

    bleu: Optional[float]
    rouge: Optional[Dict[str, float]]
    perplexity: Optional[float]
    robustness: Optional[Dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "bleu": self.bleu,
            "rouge": self.rouge,
            "perplexity": self.perplexity,
            "robustness": self.robustness,
        }


def evaluate_batch(
    generated_outputs: Sequence[str],
    references: Sequence[str],
    model_tokenizer: Optional[tuple] = None,
    generator_fn: Optional[Callable[[str], Sequence[str]]] = None,
) -> EvaluationResult:
    """Compute multiple evaluation metrics over a batch of outputs."""
    bleu = compute_bleu(generated_outputs, references)
    rouge = compute_rouge(generated_outputs, references)
    perplexity = compute_perplexity(generated_outputs, model_tokenizer)
    robustness = None

    if generator_fn:
        adversarial = generate_adversarial_prompts(references[: min(10, len(references))])
        robustness = robustness_test(generator_fn, adversarial)

    result = EvaluationResult(bleu=bleu, rouge=rouge, perplexity=perplexity, robustness=robustness)
    logger.info("Batch evaluation complete", extra=result.to_dict())
    return result


def save_evaluation_report(data: Mapping, path: Optional[Path] = None) -> Path:
    """Persist evaluation metrics to disk."""
    target = path or (EVALUATION_DIR / f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with target.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, ensure_ascii=False, indent=2)
    logger.debug("Saved evaluation report", extra={"path": str(target)})
    return target
