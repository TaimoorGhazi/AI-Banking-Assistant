"""Prompt injection detection utilities for SecureBank AI Assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from src.AI_Banking_Assistant.core.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RULES_PATH = PROJECT_ROOT / "src" / "AI_Banking_Assistant" / "guardrails" / "rules.yaml"


@dataclass(frozen=True)
class PromptInjectionFinding:
	pattern: str
	matched_text: str
	start: int
	end: int
	severity: str


DEFAULT_INJECTION_PATTERNS: Sequence[Dict[str, str]] = (
	{"pattern": r"ignore\s+previous\s+instructions", "severity": "high"},
	{"pattern": r"ignore\s+all\s+prior\s+instructions", "severity": "high"},
	{"pattern": r"disregard\s+the\s+above", "severity": "high"},
	{"pattern": r"you\s+are\s+now\s+(?:a|an)?", "severity": "medium"},
	{"pattern": r"act\s+as\s+(?:a|an)?", "severity": "medium"},
	{"pattern": r"system\s+prompt", "severity": "high"},
	{"pattern": r"reveal\s+your\s+(?:system|developer)\s+instructions", "severity": "high"},
	{"pattern": r"show\s+me\s+your\s+prompt", "severity": "high"},
	{"pattern": r"print\s+your\s+instructions", "severity": "high"},
	{"pattern": r"bypass\s+safety", "severity": "high"},
	{"pattern": r"jailbreak", "severity": "high"},
	{"pattern": r"prompt\s+injection", "severity": "high"},
	{"pattern": r"developer\s+mode", "severity": "high"},
	{"pattern": r"do\s+anything\s+now", "severity": "high"},
	{"pattern": r"as\s+an\s+ai\s+language\s+model", "severity": "medium"},
	{"pattern": r"forget\s+previous\s+context", "severity": "high"},
	{"pattern": r"roleplay\s+as", "severity": "medium"},
	{"pattern": r"override\s+your\s+instructions", "severity": "high"},
	{"pattern": r"tell\s+me\s+the\s+hidden\s+prompt", "severity": "high"},
)


def load_injection_patterns(rules_path: Optional[str] = None) -> List[Dict[str, str]]:
	"""Load injection patterns from rules.yaml with defaults as fallback."""
	path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
	if not path.exists():
		return list(DEFAULT_INJECTION_PATTERNS)

	try:
		with open(path, "r", encoding="utf-8") as handle:
			rules = yaml.safe_load(handle) or {}
	except Exception:
		logger.exception("Failed to load guardrail rules, using defaults")
		return list(DEFAULT_INJECTION_PATTERNS)

	patterns = rules.get("jailbreak_patterns") or []
	if not patterns:
		return list(DEFAULT_INJECTION_PATTERNS)

	loaded: List[Dict[str, str]] = []
	for entry in patterns:
		if isinstance(entry, str):
			loaded.append({"pattern": entry, "severity": "high"})
		elif isinstance(entry, dict) and entry.get("pattern"):
			loaded.append(
				{
					"pattern": str(entry["pattern"]),
					"severity": str(entry.get("severity", "high")),
				}
			)

	return loaded or list(DEFAULT_INJECTION_PATTERNS)


class PromptInjectionDetector:
	"""Rule-based detector for jailbreak and prompt injection attempts."""

	def __init__(self, rules_path: Optional[str] = None):
		self.patterns = load_injection_patterns(rules_path)

	def detect(self, text: str) -> List[PromptInjectionFinding]:
		"""Return all matching injection findings in the supplied text."""
		if not text or not text.strip():
			return []

		findings: List[PromptInjectionFinding] = []
		lowered = text.lower()

		for item in self.patterns:
			pattern = item["pattern"]
			severity = item.get("severity", "high")
			for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
				findings.append(
					PromptInjectionFinding(
						pattern=pattern,
						matched_text=text[match.start() : match.end()],
						start=match.start(),
						end=match.end(),
						severity=severity,
					)
				)

		return findings

	def is_injection(self, text: str) -> bool:
		"""Return True when the text contains a prompt injection attempt."""
		return bool(self.detect(text))

