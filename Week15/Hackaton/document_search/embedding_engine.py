"""Embedding generation engine using sentence-transformers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from config import BATCH_SIZE, EMBEDDINGS_DIR, EMBEDDING_MODEL

# Optional dependencies with graceful degradation
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class EmbeddingEngine:
    """
    Handles embedding generation using sentence-transformers.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = None

    def load_model(self, logger=None) -> None:
        """
        Load the sentence-transformers model.

        Args:
            logger: Optional logger instance
        """
        if self.model is None:
            if logger:
                logger.info(f"Loading embedding model: {self.model_name}")

            # Force CPU usage to avoid CUDA issues
            self.model = SentenceTransformer(self.model_name, device='cpu')

            if logger:
                logger.info(
                    f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
                )

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True,
        logger=None,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch
            show_progress: Whether to show progress bar
            logger: Optional logger instance

        Returns:
            NumPy array of embeddings with shape (num_texts, embedding_dim)
        """
        if self.model is None:
            self.load_model(logger)

        if logger:
            logger.info(
                f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})"
            )

        # Use tqdm if available and requested
        if show_progress and HAS_TQDM:
            texts_iter = tqdm(texts, desc="Embedding texts")
        else:
            texts_iter = texts

        # Generate embeddings with batching
        embeddings = self.model.encode(
            list(texts_iter),
            batch_size=batch_size,
            show_progress_bar=False,  # We handle progress ourselves
            convert_to_numpy=True,
        )

        if logger:
            logger.info(f"Generated embeddings: shape {embeddings.shape}")

        return embeddings

    def embed_chunks(
        self, chunks: list[dict], batch_size: int = BATCH_SIZE, logger=None
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' key
            batch_size: Batch size for embedding generation
            logger: Optional logger instance

        Returns:
            Tuple of (embeddings array, metadata list)
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size, logger=logger)

        # Create metadata for each embedding
        metadata = [
            {
                "doc_id": chunk.get("doc_id", "unknown"),
                "chunk_idx": chunk.get("chunk_idx", idx),
                "text": chunk["text"],
                "token_count": chunk.get("token_count", len(chunk["text"].split())),
            }
            for idx, chunk in enumerate(chunks)
        ]

        return embeddings, metadata


def save_embeddings(
    embeddings: np.ndarray,
    metadata: list[dict],
    output_name: str = "embeddings",
    logger=None,
) -> tuple[Path, Path]:
    """
    Save embeddings and metadata to disk.

    Args:
        embeddings: NumPy array of embeddings
        metadata: List of metadata dictionaries
        output_name: Base name for output files
        logger: Optional logger instance

    Returns:
        Tuple of (embeddings path, metadata path)
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = EMBEDDINGS_DIR / f"{output_name}.npy"
    metadata_path = EMBEDDINGS_DIR / f"{output_name}_metadata.json"

    # Save embeddings as NumPy array
    np.save(embeddings_path, embeddings)

    # Save metadata as JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if logger:
        logger.info(
            f"Saved {len(embeddings)} embeddings to {embeddings_path.name} "
            f"({embeddings.nbytes / 1024 / 1024:.2f} MB)"
        )

    return embeddings_path, metadata_path


def load_embeddings(output_name: str = "embeddings") -> tuple[np.ndarray, list[dict]]:
    """
    Load embeddings and metadata from disk.

    Args:
        output_name: Base name of the embedding files

    Returns:
        Tuple of (embeddings array, metadata list)

    Raises:
        FileNotFoundError: If embedding files don't exist
    """
    embeddings_path = EMBEDDINGS_DIR / f"{output_name}.npy"
    metadata_path = EMBEDDINGS_DIR / f"{output_name}_metadata.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata
