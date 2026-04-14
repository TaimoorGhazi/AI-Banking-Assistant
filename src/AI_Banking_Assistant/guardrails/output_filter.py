"""Output guardrail for SecureBank AI Assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.guardrails.pii_detector import anonymize_pii, detect_pii

logger = get_logger(__name__)


@dataclass
class OutputFilterResult:
	text: str
	sanitized: bool
	detected_entities: List[str] = field(default_factory=list)


class OutputFilter:
	"""Sanitize LLM responses before delivery."""

	def _cleanup_response(self, text: str) -> str:
		"""Remove prompt/template artifacts that should never be shown to users."""
		if not text:
			return ""

		lines = []
		for raw_line in text.splitlines():
			line = raw_line.strip()
			if not line:
				lines.append("")
				continue

			lower = line.lower()
			if lower.startswith("customer question:"):
				continue
			if lower == "answer:":
				continue
			if lower.startswith("answer:"):
				content = line.split(":", 1)[1].strip()
				if content:
					lines.append(content)
				continue
			if lower.startswith("fallback message:"):
				content = line.split(":", 1)[1].strip().strip('"')
				if content:
					lines.append(content)
				continue

			lines.append(raw_line)

		cleaned = "\n".join(lines).strip()
		return cleaned

	def sanitize(self, text: str) -> OutputFilterResult:
		if text is None:
			text = ""

		text = self._cleanup_response(text)

		detections = detect_pii(text)
		if not detections:
			return OutputFilterResult(text=text, sanitized=False, detected_entities=[])

		sanitized_text, _ = anonymize_pii(text)
		entity_types = sorted({entity.entity_type for entity in detections})
		logger.info("Sanitized LLM output containing PII: %s", ", ".join(entity_types))
		return OutputFilterResult(text=sanitized_text, sanitized=True, detected_entities=entity_types)

