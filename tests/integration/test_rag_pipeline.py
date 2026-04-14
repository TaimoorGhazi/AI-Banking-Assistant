from src.AI_Banking_Assistant.agents.orchestrator import AgentOrchestrator
from src.AI_Banking_Assistant.agents.rag_agent import RAGAgent
from src.AI_Banking_Assistant.agents.response_agent import ResponseAgent


class StubRetriever:
	def retrieve(self, query, top_k=None):
		return [{"text": "Loan approvals depend on income and credit checks.", "source": "loans.txt", "score": 0.05}]

	def format_context(self, results, max_length=2000):
		result = results[0]
		return f"[Source: {result['source']}]\n{result['text']}"


def test_end_to_end_agent_pipeline_returns_final_response():
	orchestrator = AgentOrchestrator(
		rag_agent=RAGAgent(retriever=StubRetriever()),
		response_agent=ResponseAgent(
			generator=lambda query, context, system_prompt=None, max_tokens=None: (
				f"Based on the documents, {query.lower()} -> {context}"
			)
		),
	)

	state = orchestrator.invoke("How are loan approvals decided?")

	assert state["blocked"] is False
	assert "loan approvals" in state["final_response"].lower()
	assert "[Source: loans.txt]" in state["context"]
