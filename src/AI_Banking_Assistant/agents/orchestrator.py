"""LangGraph orchestrator for SecureBank AI Assistant."""

from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.AI_Banking_Assistant.agents.guardrail_agent import GuardrailAgent
from src.AI_Banking_Assistant.agents.rag_agent import RAGAgent
from src.AI_Banking_Assistant.agents.response_agent import ResponseAgent


class ConversationState(TypedDict, total=False):
	query: str
	history: str
	top_k: int
	max_context_length: int
	max_tokens: int
	system_prompt: str
	blocked: bool
	block_reason: str
	guardrail_decision: Dict[str, Any]
	retrieval_results: list[Dict[str, Any]]
	context: str
	sources: list[str]
	draft_response: str
	final_response: str
	output_sanitized: bool
	output_filter_result: Dict[str, Any]


class AgentOrchestrator:
	"""Coordinate the guardrail, retrieval, and response agents."""

	def __init__(
		self,
		guardrail_agent: GuardrailAgent | None = None,
		rag_agent: RAGAgent | None = None,
		response_agent: ResponseAgent | None = None,
	):
		self.guardrail_agent = guardrail_agent or GuardrailAgent()
		self.rag_agent = rag_agent or RAGAgent()
		self.response_agent = response_agent or ResponseAgent()
		self.graph = self._build_graph()

	def _build_graph(self):
		workflow = StateGraph(ConversationState)
		workflow.add_node("guardrails", self.guardrail_agent.inspect_input)
		workflow.add_node("blocked_final", self._blocked_final)
		workflow.add_node("rag", self.rag_agent.run)
		workflow.add_node("generate_response", self.response_agent.run)
		workflow.add_node("output_guard", self.guardrail_agent.sanitize_output)

		workflow.set_entry_point("guardrails")
		workflow.add_conditional_edges(
			"guardrails",
			self._route_after_guardrails,
			{
				"blocked": "blocked_final",
				"continue": "rag",
			},
		)
		workflow.add_edge("blocked_final", END)
		workflow.add_edge("rag", "generate_response")
		workflow.add_edge("generate_response", "output_guard")
		workflow.add_edge("output_guard", END)
		return workflow.compile()

	def invoke(self, query: str, **state: Any) -> Dict[str, Any]:
		initial_state: ConversationState = {"query": query, **state}
		return dict(self.graph.invoke(initial_state))

	def _route_after_guardrails(self, state: ConversationState) -> Literal["blocked", "continue"]:
		return "blocked" if state.get("blocked") else "continue"

	def _blocked_final(self, state: ConversationState) -> Dict[str, Any]:
		updated_state = dict(state)
		updated_state["final_response"] = updated_state.get("final_response") or self.guardrail_agent.safe_fallback
		return updated_state

