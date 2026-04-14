"""Response generation agent for SecureBank AI Assistant."""

from __future__ import annotations

from typing import Any, Callable, Dict

from src.AI_Banking_Assistant.agents.base_agent import BaseAgent
from src.AI_Banking_Assistant.llm.inference import generate_with_context


class ResponseAgent(BaseAgent):
	"""Generate the final assistant response from retrieved context."""

	def __init__(self, generator: Callable[..., str] | None = None):
		super().__init__(name="response_agent")
		self.generator = generator or generate_with_context

	def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		if state.get("blocked"):
			return dict(state)

		query = state.get("query", "")
		context = state.get("context", "")
		history = state.get("history", "")
		system_prompt = state.get("system_prompt")
		max_tokens = state.get("max_tokens")

		response = self.generator(
			query=query,
			context=context,
			history=history,
			system_prompt=system_prompt,
			max_tokens=max_tokens,
		)

		updated_state = dict(state)
		updated_state["draft_response"] = response
		updated_state["final_response"] = response
		return updated_state
