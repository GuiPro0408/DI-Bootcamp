"""FAISS vector store for semantic search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import EMBEDDING_DIM, TOP_K, VECTOR_STORE_DIR

# Optional dependencies with graceful degradation
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM):
        """
        Initialize the vector store.

        Args:
            dimension: Dimension of the embedding vectors

        Raises:
            ImportError: If FAISS is not installed
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss-cpu not installed. Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index = None
        self.metadata = []

    def build_index(self, logger=None) -> None:
        """
        Build a new FAISS index using IndexFlatL2 (CPU-friendly).

        Args:
            logger: Optional logger instance
        """
        # Use IndexFlatL2 for CPU - simple and accurate
        self.index = faiss.IndexFlatL2(self.dimension)

        if logger:
            logger.info(f"Created FAISS IndexFlatL2 with dimension {self.dimension}")

    def add_documents(
        self, embeddings: np.ndarray, metadata: list[dict], logger=None
    ) -> None:
        """
        Add document embeddings to the index.

        Args:
            embeddings: NumPy array of embeddings (num_docs, dimension)
            metadata: List of metadata dictionaries for each embedding
            logger: Optional logger instance
        """
        if self.index is None:
            self.build_index(logger)

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        self.metadata.extend(metadata)

        if logger:
            logger.info(
                f"Added {len(embeddings)} documents to vector store (total: {self.index.ntotal})"
            )

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = TOP_K, logger=None
    ) -> list[dict]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            logger: Optional logger instance

        Returns:
            List of result dictionaries with 'text', 'doc_id', 'similarity_score', etc.
        """
        if self.index is None or self.index.ntotal == 0:
            if logger:
                logger.warning("Vector store is empty")
            return []

        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(
            query_embedding, min(top_k, self.index.ntotal)
        )

        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = {
                    **self.metadata[idx],
                    "similarity_score": float(
                        1 / (1 + distance)
                    ),  # Convert L2 distance to similarity
                    "distance": float(distance),
                }
                results.append(result)

        if logger:
            logger.info(f"Retrieved {len(results)} similar chunks")

        return results

    def save_index(self, index_name: str = "index", logger=None) -> tuple[Path, Path]:
        """
        Save the FAISS index and metadata to disk.

        Args:
            index_name: Base name for the index files
            logger: Optional logger instance

        Returns:
            Tuple of (index path, metadata path)
        """
        if self.index is None:
            raise ValueError("No index to save. Build or load an index first.")

        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        index_path = VECTOR_STORE_DIR / f"{index_name}.faiss"
        metadata_path = VECTOR_STORE_DIR / f"{index_name}_metadata.json"

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        if logger:
            logger.info(
                f"Saved vector store to {index_path.name} ({self.index.ntotal} vectors)"
            )

        return index_path, metadata_path

    def load_index(self, index_name: str = "index", logger=None) -> None:
        """
        Load a FAISS index and metadata from disk.

        Args:
            index_name: Base name of the index files
            logger: Optional logger instance

        Raises:
            FileNotFoundError: If index files don't exist
        """
        index_path = VECTOR_STORE_DIR / f"{index_name}.faiss"
        metadata_path = VECTOR_STORE_DIR / f"{index_name}_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if logger:
            logger.info(
                f"Loaded vector store from {index_path.name} ({self.index.ntotal} vectors)"
            )


def build_vector_store(
    embeddings: np.ndarray, metadata: list[dict], index_name: str = "index", logger=None
) -> VectorStore:
    """
    Build and save a vector store from embeddings.

    Args:
        embeddings: NumPy array of embeddings
        metadata: List of metadata dictionaries
        index_name: Name for the saved index
        logger: Optional logger instance

    Returns:
        Initialized VectorStore instance
    """
    store = VectorStore()
    store.build_index(logger)
    store.add_documents(embeddings, metadata, logger)
    store.save_index(index_name, logger)
    return store


def load_vector_store(index_name: str = "index", logger=None) -> VectorStore:
    """
    Load an existing vector store from disk.

    Args:
        index_name: Name of the saved index
        logger: Optional logger instance

    Returns:
        Loaded VectorStore instance
    """
    store = VectorStore()
    store.load_index(index_name, logger)
    return store
