"""Streamlit web interface for document search and summarization."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from config import UPLOADS_DIR, TOP_K
from pipeline import process_new_document_incremental
from embedding_engine import EmbeddingEngine
from vector_store import load_vector_store
from summarization import Summarizer
from logging_utils import configure_logging


# Page configuration
st.set_page_config(
    page_title="AI Document Search & Summarization", page_icon="🔍", layout="wide"
)


@st.cache_resource
def get_embedding_engine():
    """Load and cache the embedding engine."""
    logger = configure_logging()
    engine = EmbeddingEngine()
    engine.load_model(logger)
    return engine


@st.cache_resource
def get_vector_store():
    """Load and cache the vector store."""
    logger = configure_logging()
    try:
        store = load_vector_store(logger=logger)
        return store
    except FileNotFoundError:
        return None


@st.cache_resource
def get_summarizer():
    """Load and cache the summarizer."""
    logger = configure_logging()
    summarizer = Summarizer()
    summarizer.load_model(logger)
    return summarizer


def display_search_result(result: dict, idx: int):
    """
    Display a single search result in an expandable card.

    Args:
        result: Search result dictionary
        idx: Result index for display
    """
    similarity_pct = result.get("similarity_score", 0) * 100
    doc_id = result.get("doc_id", "unknown")
    chunk_idx = result.get("chunk_idx", 0)

    with st.expander(
        f"📄 Result {idx + 1}: {doc_id} (Chunk {chunk_idx}) - {similarity_pct:.1f}% match",
        expanded=(idx == 0),
    ):
        st.markdown(result.get("text", "No text available"))

        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document", doc_id)
        with col2:
            st.metric("Chunk Index", chunk_idx)
        with col3:
            st.metric("Similarity", f"{similarity_pct:.1f}%")


def main():
    """Main Streamlit application."""

    # Header
    st.title("🔍 AI-Powered Document Search & Summarization")
    st.markdown("---")

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF, Word, or text documents",
        )

        if uploaded_files:
            if st.button("Process Uploaded Documents", type="primary"):
                logger = configure_logging()
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file
                    file_path = UPLOADS_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    status_text.text(f"Processing: {uploaded_file.name}")

                    # Process document
                    result = process_new_document_incremental(file_path, logger)

                    if result["status"] == "success":
                        st.sidebar.success(f"✅ {uploaded_file.name}")
                    else:
                        st.sidebar.error(
                            f"❌ {uploaded_file.name}: {result.get('message', 'Error')}"
                        )

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("All documents processed!")
                st.success("Documents processed successfully! Clear cache to use them.")

                # Clear caches to reload vector store
                st.cache_resource.clear()

        st.markdown("---")

        # System info
        st.header("ℹ️ System Info")
        vector_store = get_vector_store()

        if vector_store and vector_store.index:
            st.metric("Total Chunks", vector_store.index.ntotal)
            st.metric(
                "Documents Processed",
                len(set(m.get("doc_id", "unknown") for m in vector_store.metadata)),
            )
        else:
            st.info("No vector store available. Upload and process documents first.")

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Settings"):
            top_k = st.slider(
                "Number of results", min_value=1, max_value=20, value=TOP_K
            )
            enable_summary = st.checkbox("Enable summarization", value=True)

    # Main search interface
    vector_store = get_vector_store()

    if (
        vector_store is None
        or vector_store.index is None
        or vector_store.index.ntotal == 0
    ):
        st.warning("⚠️ No documents indexed yet. Upload documents using the sidebar.")
        st.info(
            "**Getting Started:**\n"
            "1. Upload PDF, Word, or text documents using the sidebar\n"
            "2. Click 'Process Uploaded Documents'\n"
            "3. Start searching your documents!"
        )
        return

    # Search bar
    query = st.text_input(
        "🔎 Search your documents",
        placeholder="Enter your search query...",
        help="Enter a question or keywords to search across all indexed documents",
    )

    search_button = st.button("Search", type="primary")

    if search_button and query:
        with st.spinner("Searching..."):
            start_time = time.perf_counter()

            # Get embedding engine and generate query embedding
            embedding_engine = get_embedding_engine()
            query_embedding = embedding_engine.embed_texts([query], show_progress=False)

            # Search
            results = vector_store.search_similar(
                query_embedding[0], top_k=top_k if "top_k" in locals() else TOP_K
            )

            search_time = time.perf_counter() - start_time

        # Display results
        if results:
            st.success(f"Found {len(results)} results in {search_time:.2f} seconds")

            # Display each result
            st.markdown("### 📋 Search Results")
            for idx, result in enumerate(results):
                display_search_result(result, idx)

            # Summarization option
            if (
                st.session_state.get("enable_summary", True)
                if "enable_summary" in locals()
                else True
            ):
                st.markdown("---")
                st.markdown("### 📝 Generate Summary")

                if st.button("Summarize Top Results"):
                    with st.spinner("Generating summary..."):
                        summarizer = get_summarizer()
                        summary_result = summarizer.summarize_chunks(results[:5])

                    st.markdown("#### Summary:")
                    st.info(summary_result["summary"])

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Summarized", summary_result["num_chunks"])
                    with col2:
                        st.metric("Input Tokens", summary_result["input_length"])
                    with col3:
                        st.metric("Summary Tokens", summary_result["summary_length"])

                    st.caption(
                        f"Compression ratio: {summary_result['compression_ratio']:.1%}"
                    )
        else:
            st.warning("No results found. Try a different query.")

    elif search_button:
        st.error("Please enter a search query.")

    # Footer
    st.markdown("---")
    st.caption(
        "🤖 Powered by sentence-transformers (all-MiniLM-L6-v2), "
        "FAISS, and BART | Week 15 Hackathon Project"
    )


if __name__ == "__main__":
    main()
