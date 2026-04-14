"""Shared chat schemas and helpers."""

from __future__ import annotations

from typing import List

from .request import ChatMessage


def format_history(messages: List[ChatMessage]) -> str:
	"""Format chat messages into a plain-text conversation history."""
	if not messages:
		return ""

	lines = []
	for message in messages:
		speaker = message.role.capitalize()
		lines.append(f"{speaker}: {message.content}")
	return "\n".join(lines)

