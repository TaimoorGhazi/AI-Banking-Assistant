"""Common primitives for the SecureBank LangGraph agent layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
	"""Abstract base class for stateful agents used by the orchestrator."""

	def __init__(self, name: str):
		self.name = name

	@abstractmethod
	def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		"""Transform and return the orchestration state."""

	def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
		return self.run(state)
