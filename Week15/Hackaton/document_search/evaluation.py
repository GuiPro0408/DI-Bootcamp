"""Evaluation metrics for search and summarization quality."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from config import EVAL_NUM_QUERIES, EVAL_TOP_K_VALUES, RANDOM_STATE, REPORTS_DIR

# Optional dependencies with graceful degradation
try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def generate_synthetic_queries(
    chunks: list[dict], num_queries: int = EVAL_NUM_QUERIES
) -> list[dict]:
    """
    Generate synthetic test queries by extracting sentences from chunks.

    Args:
        chunks: List of chunk dictionaries
        num_queries: Number of queries to generate

    Returns:
        List of query dictionaries with 'query', 'source_doc_id', 'source_chunk_idx'
    """
    random.seed(RANDOM_STATE)

    # Extract sentences from random chunks
    queries = []
    sampled_chunks = random.sample(chunks, min(num_queries, len(chunks)))

    for chunk in sampled_chunks:
        # Split into sentences (simple regex-based)
        sentences = re.split(r"[.!?]+", chunk["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if sentences:
            # Take a random sentence as query
            query_text = random.choice(sentences)
            queries.append(
                {
                    "query": query_text,
                    "source_doc_id": chunk.get("doc_id", "unknown"),
                    "source_chunk_idx": chunk.get("chunk_idx", 0),
                }
            )

    return queries


def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    Calculate precision@k.

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Precision score
    """
    if not retrieved_ids or k == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    num_relevant = sum(1 for rid in top_k if rid in relevant_set)
    return num_relevant / min(k, len(top_k))


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    Calculate recall@k.

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of relevant chunk IDs
        k: Number of top results to consider

    Returns:
        Recall score
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    num_relevant = sum(1 for rid in top_k if rid in relevant_set)
    return num_relevant / len(relevant_set)


def compute_rouge_l(summary: str, reference: str) -> dict:
    """
    Compute ROUGE-L score for summarization.

    Args:
        summary: Generated summary text
        reference: Reference summary text

    Returns:
        Dictionary with ROUGE-L scores

    Raises:
        ImportError: If rouge-score is not installed
    """
    if not HAS_ROUGE:
        raise ImportError(
            "rouge-score not installed. Install with: pip install rouge-score"
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)

    return {
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def evaluate_search(
    test_queries: list[dict],
    search_function,
    k_values: list[int] = EVAL_TOP_K_VALUES,
    logger=None,
) -> dict:
    """
    Evaluate search quality using precision@k and recall@k.

    Args:
        test_queries: List of test query dictionaries
        search_function: Function that takes query text and returns results
        k_values: List of k values to evaluate
        logger: Optional logger instance

    Returns:
        Evaluation results dictionary
    """
    if logger:
        logger.info(f"Evaluating search with {len(test_queries)} queries")

    results = {
        "num_queries": len(test_queries),
        "k_values": k_values,
        "per_query": [],
        "average_metrics": {},
    }

    for query_data in test_queries:
        query_text = query_data["query"]
        relevant_ids = [(query_data["source_doc_id"], query_data["source_chunk_idx"])]

        # Perform search
        search_results = search_function(query_text)
        retrieved_ids = [(r.get("doc_id"), r.get("chunk_idx")) for r in search_results]

        # Compute metrics for each k
        query_metrics = {"query": query_text}
        for k in k_values:
            prec = precision_at_k(retrieved_ids, relevant_ids, k)
            rec = recall_at_k(retrieved_ids, relevant_ids, k)
            query_metrics[f"precision@{k}"] = prec
            query_metrics[f"recall@{k}"] = rec

        results["per_query"].append(query_metrics)

    # Calculate averages
    for k in k_values:
        avg_prec = sum(q[f"precision@{k}"] for q in results["per_query"]) / len(
            test_queries
        )
        avg_rec = sum(q[f"recall@{k}"] for q in results["per_query"]) / len(
            test_queries
        )
        results["average_metrics"][f"precision@{k}"] = avg_prec
        results["average_metrics"][f"recall@{k}"] = avg_rec

    if logger:
        logger.info(
            f"Average precision@{k_values[0]}: {results['average_metrics'][f'precision@{k_values[0]}']:.3f}"
        )
        logger.info(
            f"Average recall@{k_values[0]}: {results['average_metrics'][f'recall@{k_values[0]}']:.3f}"
        )

    return results


def save_evaluation_results(
    results: dict, output_name: str = "evaluation", logger=None
) -> Path:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_name: Base name for output file
        logger: Optional logger instance

    Returns:
        Path to saved results file
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / f"{output_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if logger:
        logger.info(f"Saved evaluation results to {output_path.name}")

    return output_path
