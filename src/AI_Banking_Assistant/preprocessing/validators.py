"""
Data validation functions for AI Banking Assistant.
Validates documents before processing.
"""

from typing import Optional, Tuple

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import MAX_INPUT_LENGTH

logger = get_logger(__name__)


def validate_document_length(text: str, max_length: int = None) -> Tuple[bool, str]:
    """Check if document length is within acceptable bounds.
    
    Args:
        text: Document text
        max_length: Maximum allowed length (default: from constants)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not text:
        return False, "Document is empty"

    if len(text.strip()) == 0:
        return False, "Document contains only whitespace"

    if len(text) < 10:
        return False, f"Document too short ({len(text)} chars, minimum 10)"

    if max_length and len(text) > max_length:
        return False, f"Document too long ({len(text)} chars, maximum {max_length})"

    return True, f"Document length valid ({len(text)} chars)"


def validate_encoding(text: str) -> Tuple[bool, str]:
    """Check if text has valid encoding (no broken characters).
    
    Args:
        text: Document text
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        text.encode("utf-8").decode("utf-8")
        return True, "Encoding valid (UTF-8)"
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        return False, f"Encoding error: {e}"


def validate_content_quality(text: str) -> Tuple[bool, str]:
    """Basic content quality check.
    
    Args:
        text: Document text
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check for mostly non-alphanumeric content
    alpha_count = sum(1 for c in text if c.isalnum())
    total_count = len(text)

    if total_count > 0 and alpha_count / total_count < 0.3:
        return False, "Document has too few alphanumeric characters (possible binary/corrupted file)"

    # Check for repetitive content
    words = text.split()
    if len(words) > 10:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.1:
            return False, "Document has too much repetitive content"

    return True, "Content quality acceptable"


def validate_document(text: str, max_length: int = None) -> Tuple[bool, list]:
    """Run all validation checks on a document.
    
    Args:
        text: Document text
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (all_passed, list_of_results)
    """
    results = []

    checks = [
        ("length", validate_document_length(text, max_length)),
        ("encoding", validate_encoding(text)),
        ("quality", validate_content_quality(text)),
    ]

    all_passed = True
    for check_name, (is_valid, message) in checks:
        results.append({"check": check_name, "valid": is_valid, "message": message})
        if not is_valid:
            all_passed = False
            logger.warning(f"Validation failed [{check_name}]: {message}")
        else:
            logger.debug(f"Validation passed [{check_name}]: {message}")

    return all_passed, results
