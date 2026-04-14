"""RAG retrieval agent for SecureBank AI Assistant."""

from __future__ import annotations

from typing import Any, Dict

from src.AI_Banking_Assistant.agents.base_agent import BaseAgent
from src.AI_Banking_Assistant.retrieval.retriever import DocumentRetriever, get_retriever


class RAGAgent(BaseAgent):
	"""Retrieve and format context for a user query."""

	def __init__(self, retriever: DocumentRetriever | None = None):
		super().__init__(name="rag_agent")
		self.retriever = retriever or get_retriever()

	def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
		query = state.get("query", "")
		top_k = state.get("top_k")
		max_context_length = state.get("max_context_length", 2000)

		results = self.retriever.retrieve(query, top_k=top_k)
		context = self.retriever.format_context(results, max_length=max_context_length)

		updated_state = dict(state)
		updated_state["retrieval_results"] = results
		updated_state["context"] = context
		updated_state["sources"] = [result.get("source", "unknown") for result in results]
		return updated_state
