"""Security audit logging for blocked and suspicious requests."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.AI_Banking_Assistant.core.logger import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

DEFAULT_AUDIT_LOG = PROJECT_ROOT / "logs" / "security_audit.log"


@dataclass
class AuditEvent:
	event_type: str
	message: str
	severity: str = "info"
	source: str = "guardrails"
	metadata: Dict[str, Any] = field(default_factory=dict)
	timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SecurityAuditLogger:
	"""Write structured security events to a dedicated log file."""

	def __init__(self, log_file: Optional[str] = None):
		self.log_path = Path(log_file) if log_file else DEFAULT_AUDIT_LOG
		self.log_path.parent.mkdir(parents=True, exist_ok=True)
		self._logger = logging.getLogger("AI_Banking_Assistant.security_audit")
		if not self._logger.handlers:
			handler = logging.FileHandler(self.log_path, encoding="utf-8")
			handler.setFormatter(logging.Formatter("%(message)s"))
			self._logger.addHandler(handler)
			self._logger.setLevel(logging.INFO)
			self._logger.propagate = False

	def log(self, event: AuditEvent) -> None:
		self._logger.info(json.dumps(asdict(event), ensure_ascii=True))


def log_security_event(
	event_type: str,
	message: str,
	severity: str = "info",
	metadata: Optional[Dict[str, Any]] = None,
	log_file: Optional[str] = None,
) -> None:
	"""Convenience helper for writing a single audit event."""
	logger_instance = SecurityAuditLogger(log_file=log_file)
	logger_instance.log(
		AuditEvent(
			event_type=event_type,
			message=message,
			severity=severity,
			metadata=metadata or {},
		)
	)
