"""Text summarization using Hugging Face transformers."""

from __future__ import annotations

import json
from pathlib import Path

from config import (
    SUMMARY_MIN_LENGTH,
    SUMMARY_MAX_LENGTH,
    SUMMARY_INPUT_MAX_TOKENS,
    SUMMARIZATION_MODEL,
    SUMMARIES_DIR,
)

# Optional dependencies with graceful degradation
try:
    from transformers import pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class Summarizer:
    """
    Text summarization using Hugging Face transformers.
    """

    def __init__(self, model_name: str = SUMMARIZATION_MODEL):
        """
        Initialize the summarizer.

        Args:
            model_name: Name of the Hugging Face summarization model

        Raises:
            ImportError: If transformers is not installed
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )

        self.model_name = model_name
        self.pipeline = None

    def load_model(self, logger=None) -> None:
        """
        Load the summarization pipeline.

        Args:
            logger: Optional logger instance
        """
        if self.pipeline is None:
            if logger:
                logger.info(f"Loading summarization model: {self.model_name}")

            # Load with CPU-optimized settings
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=-1,  # Force CPU usage
            )

            if logger:
                logger.info("Summarization model loaded")

    def summarize_text(
        self,
        text: str,
        min_length: int = SUMMARY_MIN_LENGTH,
        max_length: int = SUMMARY_MAX_LENGTH,
        logger=None,
    ) -> dict:
        """
        Generate a summary of the input text.

        Args:
            text: Input text to summarize
            min_length: Minimum length of summary in tokens
            max_length: Maximum length of summary in tokens
            logger: Optional logger instance

        Returns:
            Dictionary with 'summary', 'input_length', 'summary_length'
        """
        if self.pipeline is None:
            self.load_model(logger)

        # Truncate input if too long
        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > SUMMARY_INPUT_MAX_TOKENS:
            if logger:
                logger.warning(
                    f"Input text too long ({len(tokens)} tokens), "
                    f"truncating to {SUMMARY_INPUT_MAX_TOKENS}"
                )
            tokens = tokens[:SUMMARY_INPUT_MAX_TOKENS]
            text = tokenizer.decode(tokens, skip_special_tokens=True)

        # Generate summary
        result = self.pipeline(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,  # Deterministic output
            truncation=True,
        )

        summary_text = result[0]["summary_text"]

        # Count tokens
        summary_tokens = tokenizer.encode(summary_text, add_special_tokens=False)

        return {
            "summary": summary_text,
            "input_length": len(tokens),
            "summary_length": len(summary_tokens),
            "compression_ratio": len(summary_tokens) / len(tokens)
            if len(tokens) > 0
            else 0,
        }

    def summarize_chunks(
        self,
        chunks: list[dict],
        min_length: int = SUMMARY_MIN_LENGTH,
        max_length: int = SUMMARY_MAX_LENGTH,
        logger=None,
    ) -> dict:
        """
        Summarize multiple chunks by concatenating and summarizing.

        Args:
            chunks: List of chunk dictionaries with 'text' key
            min_length: Minimum summary length
            max_length: Maximum summary length
            logger: Optional logger instance

        Returns:
            Summary result dictionary
        """
        # Concatenate chunk texts
        combined_text = "\n\n".join(chunk["text"] for chunk in chunks)

        if logger:
            logger.info(
                f"Summarizing {len(chunks)} chunks ({len(combined_text)} characters)"
            )

        # Generate summary
        result = self.summarize_text(combined_text, min_length, max_length, logger)

        # Add metadata
        result["num_chunks"] = len(chunks)
        result["source_docs"] = list(
            set(chunk.get("doc_id", "unknown") for chunk in chunks)
        )

        return result


def save_summary(
    summary: dict, query: str, output_name: str = "summary", logger=None
) -> Path:
    """
    Save summary result to disk.

    Args:
        summary: Summary dictionary
        query: Original query that generated this summary
        output_name: Base name for output file
        logger: Optional logger instance

    Returns:
        Path to saved summary file
    """
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SUMMARIES_DIR / f"{output_name}.json"

    # Add query to summary
    full_result = {"query": query, **summary}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)

    if logger:
        logger.info(f"Saved summary to {output_path.name}")

    return output_path


def summarize_search_results(
    search_results: list[dict],
    query: str,
    model_name: str = SUMMARIZATION_MODEL,
    logger=None,
) -> dict:
    """
    Convenience function to summarize search results.

    Args:
        search_results: List of search result dictionaries
        query: Original search query
        model_name: Summarization model to use
        logger: Optional logger instance

    Returns:
        Summary dictionary
    """
    if not search_results:
        return {
            "summary": "No results found to summarize.",
            "num_chunks": 0,
            "input_length": 0,
            "summary_length": 0,
        }

    summarizer = Summarizer(model_name)
    return summarizer.summarize_chunks(search_results, logger=logger)
