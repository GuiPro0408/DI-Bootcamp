"""AI-Powered Document Search and Summarization System.

A modular pipeline for semantic document search and automatic summarization
using sentence-transformers, FAISS, and BART.
"""

from pipeline import run_pipeline, process_new_document_incremental

__version__ = "1.0.0"
__all__ = ["run_pipeline", "process_new_document_incremental"]
