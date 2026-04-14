"""Guardrail agent that bridges the safety layer into the LangGraph flow."""

from __future__ import annotations

from typing import Any, Dict

from src.AI_Banking_Assistant.agents.base_agent import BaseAgent
from src.AI_Banking_Assistant.core.exceptions import PromptInjectionDetected
from src.AI_Banking_Assistant.guardrails.input_filter import GuardrailDecision, InputFilter, ThreatLevel
from src.AI_Banking_Assistant.guardrails.output_filter import OutputFilter, OutputFilterResult


class GuardrailAgent(BaseAgent):
	"""Run input and output guardrails as part of the agent pipeline."""

	def __init__(
		self,
		input_filter: InputFilter | None = None,
		output_filter: OutputFilter | None = None,
		safe_fallback: str = "I cannot safely process this request.",
	):
		super().__init__(name="guardrail_agent")
		self.input_filter = input_filter or InputFilter()
		self.output_filter = output_filter or OutputFilter()
		self.safe_fallback = safe_fallback

	def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		return self.inspect_input(state)

	def inspect_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
		query = state.get("query", "")

		try:
			decision = self.input_filter.check(query)
		except PromptInjectionDetected as exc:
			return self._blocked_state(
				state,
				GuardrailDecision(
					threat_level=ThreatLevel.BLOCKED,
					allowed=False,
					reason=str(exc),
					metadata={"query_length": len(query)},
				),
			)

		updated_state = dict(state)
		updated_state["guardrail_decision"] = decision
		updated_state["blocked"] = not decision.allowed
		updated_state["block_reason"] = decision.reason if updated_state["blocked"] else ""
		return updated_state

	def sanitize_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
		if state.get("blocked"):
			return self._blocked_output_state(state)

		response = state.get("draft_response", "")
		result = self.output_filter.sanitize(response)

		updated_state = dict(state)
		updated_state["output_filter_result"] = result
		updated_state["final_response"] = result.text
		updated_state["output_sanitized"] = result.sanitized
		return updated_state

	def _blocked_state(self, state: Dict[str, Any], decision: GuardrailDecision) -> Dict[str, Any]:
		updated_state = dict(state)
		updated_state["guardrail_decision"] = decision
		updated_state["blocked"] = True
		updated_state["block_reason"] = decision.reason
		updated_state["final_response"] = self.safe_fallback
		updated_state["draft_response"] = self.safe_fallback
		updated_state["output_sanitized"] = False
		return updated_state

	def _blocked_output_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
		updated_state = dict(state)
		updated_state["final_response"] = updated_state.get("final_response") or self.safe_fallback
		updated_state["output_sanitized"] = False
		return updated_state
