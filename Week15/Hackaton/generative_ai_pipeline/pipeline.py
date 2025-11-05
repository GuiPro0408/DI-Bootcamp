"""Pipeline orchestration for the generative AI system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# Suppress tokenizers parallelism warning before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import data_io
from config import BATCH_SIZE, RANDOM_STATE, TEXT_MODEL, ALT_TEXT_MODEL
from ethical_filter import EthicalFilter, FilterResult
from evaluation import EvaluationResult, evaluate_batch, save_evaluation_report
from logging_utils import configure_logging
from quality_control import QualityChecker, QualityResult
from text_generation import GenerationResult, TextGenerator

logger = configure_logging(__name__)

try:
    from vae_image_gen import VAEGenerator

    HAS_VAE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_VAE = False
    VAEGenerator = None  # type: ignore


@dataclass
class PipelineArtifacts:
    """Track artifact paths generated during the pipeline run."""

    generated_texts: List[str] = field(default_factory=list)
    quality_reports: List[str] = field(default_factory=list)
    ethical_flags: List[str] = field(default_factory=list)
    evaluation_reports: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


@dataclass
class PipelineResults:
    """Summary statistics returned by pipeline execution."""

    num_generated: int
    num_passed_quality: int
    num_flagged: int
    metrics: Dict[str, float]
    artifacts: PipelineArtifacts


def _init_quality_checker() -> Optional[QualityChecker]:
    """Attempt to instantiate the quality checker."""
    try:
        return QualityChecker()
    except ImportError as exc:
        logger.warning("QualityChecker unavailable", extra={"reason": str(exc)})
        return None


def _init_ethical_filter() -> Optional[EthicalFilter]:
    """Attempt to instantiate the ethical filter."""
    try:
        return EthicalFilter()
    except ImportError as exc:
        logger.warning("EthicalFilter unavailable", extra={"reason": str(exc)})
        return None


def _init_text_generators(models: Sequence[str]) -> Dict[str, TextGenerator]:
    """Create text generator instances for requested models."""
    generators: Dict[str, TextGenerator] = {}
    for model in models:
        try:
            generators[model] = TextGenerator(default_model=model, alt_model=model)
        except ImportError as exc:
            logger.error("Failed to initialize text generator", extra={"model": model, "reason": str(exc)})
    return generators


def run_pipeline(
    prompts: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    enable_vae: bool = False,
    run_evaluation: bool = True,
) -> PipelineResults:
    """Execute the end-to-end pipeline and return summarized results."""
    models = list(models) if models else [TEXT_MODEL, ALT_TEXT_MODEL]
    generators = _init_text_generators(models)
    quality_checker = _init_quality_checker()
    ethical_filter = _init_ethical_filter()

    dataset = data_io.load_imdb_subset(split="train", sample_pct=0.05, stratify=True, seed=RANDOM_STATE)
    references = dataset.references
    prompt_source = prompts or dataset.prompts[:20]

    artifacts = PipelineArtifacts()
    all_generations: List[GenerationResult] = []
    passed_quality: List[GenerationResult] = []
    ethical_flags: List[FilterResult] = []
    quality_reports: List[QualityResult] = []

    for model_name, generator in generators.items():
        logger.info("Running text generation", extra={"model": model_name})
        generations = generator.batch_generate(prompt_source, batch_size=BATCH_SIZE, model_name=model_name)
        save_path = generator.save_results(generations, model_name=model_name)
        artifacts.generated_texts.append(str(save_path))
        all_generations.extend(generations)
        logger.info("[FLOW] Stage complete", extra={"stage": "text_generation", "model": model_name})

    if quality_checker:
        logger.info("Running quality control checks")
        for generation in all_generations:
            quality_result = quality_checker.validate_output(generation.prompt, generation.generated_text)
            report_path = quality_checker.save_report(quality_result)
            artifacts.quality_reports.append(str(report_path))
            quality_reports.append(quality_result)
            if quality_result.passed:
                passed_quality.append(generation)
        logger.info("[FLOW] Stage complete", extra={"stage": "quality_control", "passed": len(passed_quality)})
    else:
        passed_quality = list(all_generations)

    if ethical_filter:
        logger.info("Running ethical filtering")
        for generation in passed_quality:
            result = ethical_filter.scan_text(generation.generated_text)
            if result.is_flagged:
                path = ethical_filter.save_flagged(
                    result,
                    metadata={"prompt": generation.prompt, "model": generation.model_name},
                )
                artifacts.ethical_flags.append(str(path))
                ethical_flags.append(result)
        logger.info("[FLOW] Stage complete", extra={"stage": "ethical_filter", "flagged": len(ethical_flags)})

    if enable_vae and HAS_VAE:
        try:
            vae = VAEGenerator()
            if not vae.load_checkpoint():
                logger.info("Training VAE from scratch")
                vae.train_vae()
            image_paths = vae.generate(num_samples=8)
            artifacts.images.extend(str(path) for path in image_paths)
            logger.info("[FLOW] Stage complete", extra={"stage": "vae_generation", "images": len(image_paths)})
        except ImportError as exc:
            logger.warning("Skipping VAE stage", extra={"reason": str(exc)})
    elif enable_vae and not HAS_VAE:
        logger.warning("VAE dependencies unavailable; skipping image generation stage.")

    metrics: Dict[str, float] = {}
    evaluation_result: Optional[EvaluationResult] = None
    if run_evaluation and quality_checker:
        logger.info("Running evaluation stage")
        model_tokenizer = None
        try:
            model_tokenizer = quality_checker.load_perplexity_model()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Perplexity model unavailable for evaluation", extra={"reason": str(exc)})

        generated_texts = [result.generated_text for result in passed_quality]
        ref_subset = references[: len(generated_texts)]
        evaluation_result = evaluate_batch(
            generated_outputs=generated_texts,
            references=ref_subset,
            model_tokenizer=model_tokenizer,
        )
        evaluation_path = save_evaluation_report(evaluation_result.to_dict())
        artifacts.evaluation_reports.append(str(evaluation_path))
        metrics = {}
        for key, value in evaluation_result.to_dict().items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metrics[f"{key}_{sub_key}"] = sub_value
        logger.info("[FLOW] Stage complete", extra={"stage": "evaluation"})

    pipeline_results = PipelineResults(
        num_generated=len(all_generations),
        num_passed_quality=len(passed_quality),
        num_flagged=len(ethical_flags),
        metrics=metrics,
        artifacts=artifacts,
    )

    logger.info("Pipeline run completed", extra=pipeline_results.__dict__)
    return pipeline_results
