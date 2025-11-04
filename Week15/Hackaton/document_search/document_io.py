"""Document ingestion and text extraction utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import DIRECTORIES, RAW_TEXTS_DIR

# Optional dependencies with graceful degradation
try:
    import PyPDF2

    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import docx

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def prepare_output_structure(logger=None) -> None:
    """
    Create the nested output directory tree.

    Args:
        logger: Optional logger instance for status messages
    """
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Output directory structure prepared")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string

    Raises:
        ImportError: If PyPDF2 is not installed
        Exception: If file is corrupted or unreadable
    """
    if not HAS_PYPDF2:
        raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")

    try:
        text_parts = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF {pdf_path.name}: {str(e)}")


def extract_text_from_docx(docx_path: Path) -> str:
    """
    Extract text from a Word document using python-docx.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        Extracted text as a single string

    Raises:
        ImportError: If python-docx is not installed
        Exception: If file is corrupted or unreadable
    """
    if not HAS_DOCX:
        raise ImportError(
            "python-docx not installed. Install with: pip install python-docx"
        )

    try:
        doc = docx.Document(docx_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        return "\n".join(text_parts)

    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX {docx_path.name}: {str(e)}")


def extract_text_from_txt(txt_path: Path) -> str:
    """
    Read text from a plain text file.

    Args:
        txt_path: Path to the TXT file

    Returns:
        File contents as a string

    Raises:
        Exception: If file cannot be read
    """
    try:
        return txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        try:
            return txt_path.read_text(encoding="latin-1")
        except Exception as e:
            raise Exception(f"Failed to read text file {txt_path.name}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to read text file {txt_path.name}: {str(e)}")


def extract_text(file_path: Path) -> str:
    """
    Extract text from a document based on file extension.

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text as a string

    Raises:
        ValueError: If file type is not supported
        Exception: If extraction fails
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    elif suffix in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .doc, .txt, .md"
        )


def save_raw_text(doc_id: str, text: str, logger=None) -> Path:
    """
    Save extracted raw text to the raw_texts directory.

    Args:
        doc_id: Unique identifier for the document
        text: Extracted text content
        logger: Optional logger instance

    Returns:
        Path to the saved text file
    """
    RAW_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_TEXTS_DIR / f"{doc_id}.txt"
    output_path.write_text(text, encoding="utf-8")

    if logger:
        logger.info(f"Saved raw text for document {doc_id} ({len(text)} characters)")

    return output_path


def load_raw_text(doc_id: str) -> str:
    """
    Load previously saved raw text.

    Args:
        doc_id: Document identifier

    Returns:
        Raw text content

    Raises:
        FileNotFoundError: If the text file doesn't exist
    """
    text_path = RAW_TEXTS_DIR / f"{doc_id}.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"No raw text found for document {doc_id}")

    return text_path.read_text(encoding="utf-8")


def get_document_id(file_path: Path) -> str:
    """
    Generate a document ID from a file path.

    Args:
        file_path: Path to the document file

    Returns:
        Document ID (filename without extension)
    """
    return file_path.stem


def list_processed_documents() -> list[str]:
    """
    List all document IDs that have been processed.

    Returns:
        List of document IDs
    """
    if not RAW_TEXTS_DIR.exists():
        return []

    return [p.stem for p in RAW_TEXTS_DIR.glob("*.txt")]
