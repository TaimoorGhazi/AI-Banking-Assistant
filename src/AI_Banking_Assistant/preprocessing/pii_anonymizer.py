"""
PII detection and anonymization using Microsoft Presidio.
Identifies and replaces sensitive information in text.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import PII_PLACEHOLDER_MAP
from src.AI_Banking_Assistant.core.exceptions import PIIAnonymizationError

logger = get_logger(__name__)

# Try to import Presidio - gracefully handle if not installed
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False
    logger.warning("Presidio not installed. Using regex-based PII detection as fallback.")


# Regex patterns for fallback PII detection
REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ACCOUNT_NUMBER": r"\b\d{8,17}\b",
}


def detect_pii_presidio(text: str) -> List[Dict]:
    """Detect PII entities using Presidio analyzer.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of detected PII entities with type, start, end, score
    """
    if not HAS_PRESIDIO:
        return detect_pii_regex(text)

    try:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "US_SSN", "CREDIT_CARD", "IBAN_CODE",
                "US_BANK_NUMBER", "IP_ADDRESS",
            ],
            language="en",
        )

        detections = []
        for result in results:
            detections.append({
                "type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end],
            })

        logger.info(f"Presidio detected {len(detections)} PII entities")
        return detections

    except Exception as e:
        logger.error(f"Presidio analysis failed: {e}")
        return detect_pii_regex(text)


def detect_pii_regex(text: str) -> List[Dict]:
    """Detect PII using regex patterns (fallback method).
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of detected PII entities
    """
    detections = []

    for pii_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            detections.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "score": 0.8,
                "text": match.group(),
            })

    logger.info(f"Regex detected {len(detections)} PII entities")
    return detections


def anonymize_text(text: str) -> Tuple[str, List[Dict]]:
    """Anonymize PII in text by replacing with placeholders.
    
    Args:
        text: Input text containing PII
        
    Returns:
        Tuple of (anonymized_text, list_of_detections)
        
    Raises:
        PIIAnonymizationError: If anonymization fails
    """
    if not text or not text.strip():
        return text, []

    try:
        if HAS_PRESIDIO:
            return _anonymize_with_presidio(text)
        else:
            return _anonymize_with_regex(text)
    except Exception as e:
        raise PIIAnonymizationError(f"Failed to anonymize text: {e}")


def _anonymize_with_presidio(text: str) -> Tuple[str, List[Dict]]:
    """Anonymize using Presidio engine."""
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(
        text=text,
        entities=list(PII_PLACEHOLDER_MAP.keys()),
        language="en",
    )

    # Build operator config for each entity type
    operators = {}
    for entity_type, placeholder in PII_PLACEHOLDER_MAP.items():
        operators[entity_type] = OperatorConfig("replace", {"new_value": placeholder})

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    detections = [
        {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
        for r in results
    ]

    logger.info(f"Anonymized {len(detections)} PII entities with Presidio")
    return anonymized.text, detections


def _anonymize_with_regex(text: str) -> Tuple[str, List[Dict]]:
    """Anonymize using regex patterns (fallback)."""
    detections = detect_pii_regex(text)
    anonymized = text

    placeholder_map = {
        "EMAIL": "[EMAIL]",
        "PHONE": "[PHONE]",
        "SSN": "[SSN]",
        "CREDIT_CARD": "[CREDIT_CARD]",
        "ACCOUNT_NUMBER": "[ACCOUNT]",
    }

    # Sort by position (reverse) to replace from end to start
    sorted_detections = sorted(detections, key=lambda d: d["start"], reverse=True)

    for detection in sorted_detections:
        placeholder = placeholder_map.get(detection["type"], "[PII]")
        anonymized = (
            anonymized[:detection["start"]]
            + placeholder
            + anonymized[detection["end"]:]
        )

    logger.info(f"Anonymized {len(detections)} PII entities with regex")
    return anonymized, detections


def detect_pii(text: str) -> List[Dict]:
    """Main PII detection function. Uses Presidio if available, regex otherwise."""
    if HAS_PRESIDIO:
        return detect_pii_presidio(text)
    return detect_pii_regex(text)
