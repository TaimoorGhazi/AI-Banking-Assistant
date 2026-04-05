"""
Data ingestion module for AI Banking Assistant.
Loads documents from various file formats (txt, pdf, docx).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import SUPPORTED_FILE_EXTENSIONS, DATA_RAW_DIR
from src.AI_Banking_Assistant.core.exceptions import DataIngestionError

logger = get_logger(__name__)


def load_text_file(filepath: str) -> str:
    """Load content from a plain text file.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        File content as string
        
    Raises:
        DataIngestionError: If file cannot be read
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Loaded text file: {filepath} ({len(content)} chars)")
        return content
    except Exception as e:
        raise DataIngestionError(f"Failed to load text file {filepath}: {e}")


def load_pdf_file(filepath: str) -> str:
    """Load content from a PDF file.
    
    Args:
        filepath: Path to the .pdf file
        
    Returns:
        Extracted text content
        
    Raises:
        DataIngestionError: If file cannot be read
    """
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(filepath)
        content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text + "\n"
        logger.info(f"Loaded PDF: {filepath} ({len(reader.pages)} pages, {len(content)} chars)")
        return content
    except ImportError:
        raise DataIngestionError("PyPDF2 not installed. Run: pip install PyPDF2")
    except Exception as e:
        raise DataIngestionError(f"Failed to load PDF {filepath}: {e}")


def load_docx_file(filepath: str) -> str:
    """Load content from a Word document.
    
    Args:
        filepath: Path to the .docx file
        
    Returns:
        Extracted text content
        
    Raises:
        DataIngestionError: If file cannot be read
    """
    try:
        from docx import Document

        doc = Document(filepath)
        content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        logger.info(f"Loaded DOCX: {filepath} ({len(doc.paragraphs)} paragraphs, {len(content)} chars)")
        return content
    except ImportError:
        raise DataIngestionError("python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        raise DataIngestionError(f"Failed to load DOCX {filepath}: {e}")


def load_document(filepath: str) -> str:
    """Load a document based on its file extension.
    
    Args:
        filepath: Path to the document
        
    Returns:
        Document content as string
        
    Raises:
        DataIngestionError: If format not supported or file cannot be read
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".txt":
        return load_text_file(filepath)
    elif ext == ".pdf":
        return load_pdf_file(filepath)
    elif ext == ".docx":
        return load_docx_file(filepath)
    else:
        raise DataIngestionError(
            f"Unsupported file format: {ext}. Supported: {SUPPORTED_FILE_EXTENSIONS}"
        )


def ingest_directory(dir_path: str = None) -> List[Dict[str, str]]:
    """Load all supported documents from a directory.
    
    Args:
        dir_path: Path to directory (defaults to data/raw/)
        
    Returns:
        List of dicts with 'filename', 'filepath', and 'content' keys
        
    Raises:
        DataIngestionError: If directory does not exist
    """
    if dir_path is None:
        dir_path = DATA_RAW_DIR

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise DataIngestionError(f"Directory does not exist: {dir_path}")

    documents = []
    supported_exts = set(SUPPORTED_FILE_EXTENSIONS)

    for filepath in sorted(dir_path.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in supported_exts:
            try:
                content = load_document(str(filepath))
                documents.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "content": content,
                })
            except DataIngestionError as e:
                logger.warning(f"Skipping {filepath.name}: {e}")

    logger.info(f"Ingested {len(documents)} documents from {dir_path}")
    return documents


if __name__ == "__main__":
    # Quick test
    docs = ingest_directory()
    for doc in docs:
        print(f"  {doc['filename']}: {len(doc['content'])} chars")
