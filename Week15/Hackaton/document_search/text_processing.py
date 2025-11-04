"""Text preprocessing and chunking utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from config import CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Optional dependencies with graceful degradation
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing excessive whitespace and headers.

    Args:
        text: Raw input text

    Returns:
        Cleaned text
    """
    # Remove multiple consecutive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove common header/footer patterns (page numbers, etc.)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*Page \d+\s*\n", "\n", text, flags=re.IGNORECASE)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    model_name: Optional[str] = None,
) -> list[dict]:
    """
    Split text into overlapping chunks using tokenizer.

    Args:
        text: Input text to chunk
        chunk_size: Target number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        model_name: Optional model name for tokenizer (uses default if None)

    Returns:
        List of chunk dictionaries with 'text' and 'token_count' keys

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )

    # Load tokenizer (use default model for consistency)
    from config import EMBEDDING_MODEL

    # Force CPU to avoid CUDA issues
    model = SentenceTransformer(model_name or EMBEDDING_MODEL, device='cpu')
    tokenizer = model.tokenizer

    # Tokenize the entire text (warnings about sequence length are expected for long texts)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(tokens):
        # Get chunk tokens
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({"text": chunk_text, "token_count": len(chunk_tokens)})

        # Move to next chunk with overlap
        start += chunk_size - overlap

        # Break if we've reached the end
        if end == len(tokens):
            break

    return chunks


def save_chunks(doc_id: str, chunks: list[dict], logger=None) -> Path:
    """
    Save chunks to JSON file with metadata.

    Args:
        doc_id: Document identifier
        chunks: List of chunk dictionaries
        logger: Optional logger instance

    Returns:
        Path to the saved chunks file
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Add chunk indices
    chunks_with_metadata = [
        {
            "doc_id": doc_id,
            "chunk_idx": idx,
            "text": chunk["text"],
            "token_count": chunk["token_count"],
        }
        for idx, chunk in enumerate(chunks)
    ]

    output_path = CHUNKS_DIR / f"{doc_id}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)

    if logger:
        total_tokens = sum(c["token_count"] for c in chunks)
        logger.info(
            f"Saved {len(chunks)} chunks for document {doc_id} "
            f"({total_tokens} total tokens)"
        )

    return output_path


def load_chunks(doc_id: str) -> list[dict]:
    """
    Load previously saved chunks.

    Args:
        doc_id: Document identifier

    Returns:
        List of chunk dictionaries

    Raises:
        FileNotFoundError: If chunks file doesn't exist
    """
    chunks_path = CHUNKS_DIR / f"{doc_id}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks found for document {doc_id}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_chunks() -> list[dict]:
    """
    Load all chunks from all processed documents.

    Returns:
        List of all chunk dictionaries
    """
    if not CHUNKS_DIR.exists():
        return []

    all_chunks = []
    for chunks_file in CHUNKS_DIR.glob("*_chunks.json"):
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    return all_chunks


def process_document_text(
    doc_id: str,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    logger=None,
) -> list[dict]:
    """
    Complete text processing pipeline: clean, chunk, and save.

    Args:
        doc_id: Document identifier
        text: Raw input text
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks
        logger: Optional logger instance

    Returns:
        List of processed chunks
    """
    # Clean the text
    cleaned_text = clean_text(text)

    if logger:
        logger.info(
            f"Cleaned text for {doc_id}: {len(text)} -> {len(cleaned_text)} characters"
        )

    # Split into chunks
    chunks = split_into_chunks(cleaned_text, chunk_size, overlap)

    # Save chunks
    save_chunks(doc_id, chunks, logger)

    return chunks
