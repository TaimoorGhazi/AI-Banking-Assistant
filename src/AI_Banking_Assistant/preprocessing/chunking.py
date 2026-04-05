"""
Document chunking for AI Banking Assistant.
Splits documents into overlapping chunks for embedding and retrieval.
"""

from typing import List, Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = get_logger(__name__)


def chunk_by_characters(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into chunks by character count with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the chunk boundary
            best_break = -1
            for punct in [".\n", ".\n\n", ". ", "!\n", "? "]:
                idx = text.rfind(punct, start + chunk_size // 2, end)
                if idx > best_break:
                    best_break = idx + len(punct)

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= len(text):
            break

    logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_sentences: int = 2,
) -> List[str]:
    """Split text into chunks at sentence boundaries.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Simple sentence splitting
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap sentences
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Created {len(chunks)} sentence-based chunks from {len(sentences)} sentences")
    return chunks


def chunk_by_paragraphs(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text by paragraphs, merging small paragraphs.
    
    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.info(f"Created {len(chunks)} paragraph-based chunks")
    return chunks


def chunk_document(
    text: str,
    method: str = "sentences",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Chunk a document using the specified method.
    
    Args:
        text: Document text
        method: Chunking method - 'characters', 'sentences', or 'paragraphs'
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks (for character method)
        
    Returns:
        List of text chunks
    """
    if method == "characters":
        return chunk_by_characters(text, chunk_size, chunk_overlap)
    elif method == "sentences":
        return chunk_by_sentences(text, chunk_size)
    elif method == "paragraphs":
        return chunk_by_paragraphs(text, chunk_size)
    else:
        logger.warning(f"Unknown method '{method}', defaulting to sentences")
        return chunk_by_sentences(text, chunk_size)
