"""
Text cleaning and normalization for AI Banking Assistant.
Prepares raw document text for embedding and retrieval.
"""

import re
import unicodedata
from typing import Optional

from src.AI_Banking_Assistant.core.logger import get_logger

logger = get_logger(__name__)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into single spaces and strip lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to their standard form."""
    return unicodedata.normalize("NFKD", text)


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """Remove non-printable and control characters.
    
    Args:
        text: Input text
        keep_punctuation: If True, keeps standard punctuation
    """
    if keep_punctuation:
        text = re.sub(r"[^\w\s.,!?;:'\"\-()/@#$%&*+\n]", "", text)
    else:
        text = re.sub(r"[^\w\s]", "", text)
    return text


def normalize_line_endings(text: str) -> str:
    """Convert all line endings to unix-style (LF)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "[URL]", text)


def remove_email_addresses(text: str) -> str:
    """Replace email addresses with placeholder."""
    return re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def clean_document(
    text: str,
    do_lowercase: bool = False,
    do_remove_urls: bool = False,
    do_remove_emails: bool = False,
) -> str:
    """Apply full cleaning pipeline to document text.
    
    Args:
        text: Raw document text
        do_lowercase: Whether to lowercase the text
        do_remove_urls: Whether to replace URLs with placeholder
        do_remove_emails: Whether to replace emails with placeholder
        
    Returns:
        Cleaned text
    """
    if not text or not text.strip():
        logger.warning("Empty text received for cleaning")
        return ""

    original_len = len(text)

    # Step 1: Normalize line endings
    text = normalize_line_endings(text)

    # Step 2: Normalize unicode
    text = normalize_unicode(text)

    # Step 3: Remove special characters
    text = remove_special_characters(text)

    # Step 4: Remove extra whitespace
    text = remove_extra_whitespace(text)

    # Optional steps
    if do_remove_urls:
        text = remove_urls(text)
    if do_remove_emails:
        text = remove_email_addresses(text)
    if do_lowercase:
        text = lowercase(text)

    logger.info(f"Cleaned text: {original_len} -> {len(text)} chars")
    return text
