"""Input guardrail for SecureBank AI Assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import MAX_INPUT_LENGTH
from src.AI_Banking_Assistant.core.exceptions import PromptInjectionDetected
from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.guardrails.audit_logger import log_security_event
from src.AI_Banking_Assistant.guardrails.pii_detector import detect_pii
from src.AI_Banking_Assistant.guardrails.prompt_injection_detector import PromptInjectionDetector

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RULES_PATH = PROJECT_ROOT / "src" / "AI_Banking_Assistant" / "guardrails" / "rules.yaml"


class ThreatLevel(str, Enum):
	SAFE = "SAFE"
	SUSPICIOUS = "SUSPICIOUS"
	BLOCKED = "BLOCKED"


@dataclass
class GuardrailDecision:
	threat_level: ThreatLevel
	allowed: bool
	reason: str
	matched_patterns: List[str] = field(default_factory=list)
	detected_entities: List[str] = field(default_factory=list)
	metadata: Dict[str, object] = field(default_factory=dict)


def _load_rules(rules_path: Optional[str] = None) -> Dict[str, object]:
	path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
	if not path.exists():
		return {}

	try:
		with open(path, "r", encoding="utf-8") as handle:
			return yaml.safe_load(handle) or {}
	except Exception:
		logger.exception("Failed to read guardrail rules; continuing with defaults")
		return {}


class InputFilter:
	"""Apply guardrails to incoming user messages."""

	def __init__(self, rules_path: Optional[str] = None):
		self.rules = _load_rules(rules_path)
		self.detector = PromptInjectionDetector(rules_path=rules_path)
		self.max_input_length = int(
			self.rules.get("max_input_length")
			or Config().get("guardrails", "max_input_length", default=MAX_INPUT_LENGTH)
		)
		keywords = self.rules.get("banking_keywords") or [
			"account",
			"balance",
			"transaction",
			"transfer",
			"loan",
			"card",
			"credit",
			"debit",
			"deposit",
			"withdraw",
			"bank",
			"branch",
			"statement",
			"interest",
			"fee",
			"securebank",
			"mortgage",
			"routing",
		]
		self.domain_keywords = {keyword.lower() for keyword in keywords}

	def check(self, text: str) -> GuardrailDecision:
		"""Return a structured decision for the given message."""
		if text is None:
			text = ""

		stripped = text.strip()
		metadata = {"length": len(text)}

		if not stripped:
			return GuardrailDecision(
				threat_level=ThreatLevel.SUSPICIOUS,
				allowed=False,
				reason="Empty input is not actionable.",
				metadata=metadata,
			)

		if len(text) > self.max_input_length:
			log_security_event(
				event_type="input_too_long",
				message="User input exceeded configured length limit.",
				severity="warning",
				metadata={"length": len(text), "limit": self.max_input_length},
			)
			return GuardrailDecision(
				threat_level=ThreatLevel.BLOCKED,
				allowed=False,
				reason=f"Input exceeds maximum length of {self.max_input_length} characters.",
				metadata=metadata,
			)

		injection_findings = self.detector.detect(text)
		if injection_findings:
			patterns = sorted({finding.pattern for finding in injection_findings})
			log_security_event(
				event_type="prompt_injection_detected",
				message="Prompt injection attempt blocked.",
				severity="critical",
				metadata={"patterns": patterns, "length": len(text)},
			)
			raise PromptInjectionDetected("Prompt injection attempt detected.")

		pii_entities = detect_pii(text)
		entity_types = sorted({entity.entity_type for entity in pii_entities})
		if entity_types:
			return GuardrailDecision(
				threat_level=ThreatLevel.SUSPICIOUS,
				allowed=True,
				reason="Potential PII detected; proceed with caution.",
				detected_entities=entity_types,
				metadata=metadata,
			)

		if not self._is_banking_related(text):
			return GuardrailDecision(
				threat_level=ThreatLevel.SUSPICIOUS,
				allowed=True,
				reason="Request appears to be outside the banking domain.",
				metadata=metadata,
			)

		return GuardrailDecision(
			threat_level=ThreatLevel.SAFE,
			allowed=True,
			reason="Input passed guardrail checks.",
			metadata=metadata,
		)

	def _is_banking_related(self, text: str) -> bool:
		lowered = text.lower()
		if "securebank" in lowered:
			return True
		return any(keyword in lowered for keyword in self.domain_keywords)

