"""PII detection and anonymization helpers for guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.AI_Banking_Assistant.core.logger import get_logger

logger = get_logger(__name__)

try:
	from presidio_analyzer import AnalyzerEngine
	from presidio_anonymizer import AnonymizerEngine
	from presidio_anonymizer.entities import OperatorConfig

	HAS_PRESIDIO = True
except ImportError:
	HAS_PRESIDIO = False
	logger.warning("Presidio is not installed; using regex-based PII detection.")


@dataclass(frozen=True)
class PIIEntity:
	entity_type: str
	start: int
	end: int
	score: float
	text: str


DEFAULT_PII_PLACEHOLDERS: Dict[str, str] = {
	"PERSON": "[NAME]",
	"EMAIL_ADDRESS": "[EMAIL]",
	"PHONE_NUMBER": "[PHONE]",
	"US_SSN": "[SSN]",
	"CREDIT_CARD": "[CREDIT_CARD]",
	"IBAN_CODE": "[IBAN]",
	"US_BANK_NUMBER": "[ACCOUNT]",
	"IP_ADDRESS": "[IP_ADDRESS]",
	"LOCATION": "[ADDRESS]",
	"DATE_TIME": "[DATE_TIME]",
}

REGEX_PATTERNS: Dict[str, str] = {
	"EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
	"PHONE_NUMBER": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
	"US_SSN": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
	"CREDIT_CARD": r"\b(?:\d[ -]*?){13,19}\b",
	"US_BANK_NUMBER": r"\b\d{8,17}\b",
	"IBAN_CODE": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
	"IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
	"DATE_TIME": r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b",
}


def _normalize_entities(entities: Optional[List[str]]) -> List[str]:
	return entities or list(DEFAULT_PII_PLACEHOLDERS.keys())


def detect_pii(text: str, entities: Optional[List[str]] = None) -> List[PIIEntity]:
	"""Detect PII entities in text using Presidio or regex fallback."""
	if not text or not text.strip():
		return []

	if HAS_PRESIDIO:
		return _detect_with_presidio(text, entities)
	return _detect_with_regex(text, entities)


def anonymize_pii(text: str, entities: Optional[List[str]] = None) -> Tuple[str, List[PIIEntity]]:
	"""Replace detected PII with placeholders and return the anonymized text."""
	detections = detect_pii(text, entities)
	if not detections:
		return text, []

	anonymized = text
	for finding in sorted(detections, key=lambda item: item.start, reverse=True):
		placeholder = DEFAULT_PII_PLACEHOLDERS.get(finding.entity_type, "[PII]")
		anonymized = anonymized[: finding.start] + placeholder + anonymized[finding.end :]

	return anonymized, detections


def _detect_with_presidio(text: str, entities: Optional[List[str]]) -> List[PIIEntity]:
	analyzer = AnalyzerEngine()
	results = analyzer.analyze(text=text, entities=_normalize_entities(entities), language="en")

	findings: List[PIIEntity] = []
	for result in results:
		findings.append(
			PIIEntity(
				entity_type=result.entity_type,
				start=result.start,
				end=result.end,
				score=float(result.score),
				text=text[result.start : result.end],
			)
		)
	return findings


def _detect_with_regex(text: str, entities: Optional[List[str]]) -> List[PIIEntity]:
	requested = set(_normalize_entities(entities))
	findings: List[PIIEntity] = []

	for entity_name, pattern in REGEX_PATTERNS.items():
		if entity_name not in requested:
			continue
		for match in re.finditer(pattern, text):
			findings.append(
				PIIEntity(
					entity_type=entity_name,
					start=match.start(),
					end=match.end(),
					score=0.8,
					text=match.group(),
				)
			)

	return findings


class PIIDetector:
	"""Small convenience wrapper used by higher-level filters."""

	def detect(self, text: str, entities: Optional[List[str]] = None) -> List[PIIEntity]:
		return detect_pii(text, entities)

	def anonymize(self, text: str, entities: Optional[List[str]] = None) -> Tuple[str, List[PIIEntity]]:
		return anonymize_pii(text, entities)
