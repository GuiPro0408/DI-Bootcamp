"""Main pipeline orchestration for document search system."""

from __future__ import annotations

import warnings
from pathlib import Path

from config import UPLOADS_DIR, MAX_DOCS
from document_io import (
    prepare_output_structure,
    extract_text,
    save_raw_text,
    get_document_id,
)
from text_processing import process_document_text, load_all_chunks
from embedding_engine import EmbeddingEngine, save_embeddings, load_embeddings
from vector_store import build_vector_store, load_vector_store
from evaluation import (
    generate_synthetic_queries,
    evaluate_search,
    save_evaluation_results,
)
from logging_utils import configure_logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def process_single_document(
    file_path: Path, embedding_engine: EmbeddingEngine, logger=None
) -> tuple[str, list[dict]]:
    """
    Process a single document through the full pipeline.

    Args:
        file_path: Path to the document file
        embedding_engine: Initialized embedding engine
        logger: Optional logger instance

    Returns:
        Tuple of (document_id, chunks)
    """
    doc_id = get_document_id(file_path)

    # Extract text
    if logger:
        logger.info(f"Processing document: {file_path.name}")

    try:
        raw_text = extract_text(file_path)
        save_raw_text(doc_id, raw_text, logger)

        # Process and chunk
        chunks = process_document_text(doc_id, raw_text, logger=logger)

        return doc_id, chunks

    except Exception as e:
        if logger:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
        return doc_id, []


def run_pipeline(
    documents_dir: Path | None = None,
    run_evaluation: bool = True,
    rebuild_index: bool = True,
) -> dict:
    """
    Execute the full document search and summarization pipeline.

    Args:
        documents_dir: Directory containing documents to process (default: uploads/)
        run_evaluation: Whether to run evaluation after indexing
        rebuild_index: Whether to rebuild the index from scratch

    Returns:
        Pipeline results dictionary
    """
    logger = configure_logging()
    logger.info("=" * 60)
    logger.info("Starting Document Search & Summarization Pipeline")
    logger.info("=" * 60)

    # Prepare output structure
    prepare_output_structure(logger)

    # Determine document source
    doc_dir = documents_dir or UPLOADS_DIR
    if not doc_dir.exists():
        logger.warning(f"Documents directory not found: {doc_dir}")
        logger.info("Creating empty uploads directory. Add documents and rerun.")
        doc_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "no_documents", "message": "No documents to process"}

    # Find documents
    document_files = []
    for pattern in ["*.pdf", "*.docx", "*.doc", "*.txt", "*.md"]:
        document_files.extend(doc_dir.glob(pattern))

    if not document_files:
        logger.warning(f"No documents found in {doc_dir}")
        return {"status": "no_documents", "message": "No supported documents found"}

    # Check document limit
    if len(document_files) > MAX_DOCS:
        logger.warning(
            f"Found {len(document_files)} documents, limiting to {MAX_DOCS} for CPU constraints"
        )
        document_files = document_files[:MAX_DOCS]

    logger.info(f"Found {len(document_files)} documents to process")

    # Initialize embedding engine
    embedding_engine = EmbeddingEngine()
    embedding_engine.load_model(logger)

    # Process all documents
    all_chunks = []
    processed_docs = []

    for file_path in document_files:
        doc_id, chunks = process_single_document(file_path, embedding_engine, logger)
        if chunks:
            all_chunks.extend(chunks)
            processed_docs.append(doc_id)

    if not all_chunks:
        logger.error("No chunks generated from documents")
        return {
            "status": "error",
            "message": "Failed to extract content from documents",
        }

    logger.info(
        f"Processed {len(processed_docs)} documents into {len(all_chunks)} chunks"
    )

    # Generate embeddings
    logger.info("Generating embeddings for all chunks...")
    embeddings, metadata = embedding_engine.embed_chunks(all_chunks, logger=logger)
    save_embeddings(embeddings, metadata, logger=logger)

    # Build vector store
    logger.info("Building FAISS vector store...")
    vector_store = build_vector_store(embeddings, metadata, logger=logger)

    results = {
        "status": "success",
        "num_documents": len(processed_docs),
        "num_chunks": len(all_chunks),
        "embedding_dimension": embeddings.shape[1],
        "processed_documents": processed_docs,
    }

    # Run evaluation if requested
    if run_evaluation and len(all_chunks) >= 10:
        logger.info("Running evaluation...")

        # Generate synthetic test queries
        test_queries = generate_synthetic_queries(all_chunks)
        logger.info(f"Generated {len(test_queries)} synthetic test queries")

        # Define search function for evaluation
        def search_fn(query_text: str) -> list[dict]:
            query_embedding = embedding_engine.embed_texts(
                [query_text], show_progress=False
            )
            return vector_store.search_similar(query_embedding[0])

        # Evaluate
        eval_results = evaluate_search(test_queries, search_fn, logger=logger)
        save_evaluation_results(eval_results, logger=logger)

        results["evaluation"] = eval_results["average_metrics"]

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(
        f"Processed: {results['num_documents']} documents, {results['num_chunks']} chunks"
    )
    if "evaluation" in results:
        logger.info(
            f"Search Quality: Precision@5={results['evaluation'].get('precision@5', 0):.3f}"
        )
    logger.info("=" * 60)

    return results


def process_new_document_incremental(file_path: Path, logger=None) -> dict:
    """
    Process a new document and add it to existing index.

    Args:
        file_path: Path to the new document
        logger: Optional logger instance

    Returns:
        Processing result dictionary
    """
    if logger is None:
        logger = configure_logging()

    try:
        # Initialize embedding engine
        embedding_engine = EmbeddingEngine()
        embedding_engine.load_model(logger)

        # Process document
        doc_id, chunks = process_single_document(file_path, embedding_engine, logger)

        if not chunks:
            return {"status": "error", "message": "No chunks generated"}

        # Generate embeddings
        embeddings, metadata = embedding_engine.embed_chunks(chunks, logger=logger)

        # Load existing vector store
        try:
            vector_store = load_vector_store(logger=logger)
        except FileNotFoundError:
            # No existing index, create new one
            vector_store = build_vector_store(embeddings, metadata, logger=logger)
            return {
                "status": "success",
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "message": "Created new index",
            }

        # Add to existing index
        vector_store.add_documents(embeddings, metadata, logger)
        vector_store.save_index(logger=logger)

        return {
            "status": "success",
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "total_documents": vector_store.index.ntotal,
        }

    except Exception as e:
        if logger:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
        return {"status": "error", "message": str(e)}
