from fastapi.testclient import TestClient
from unittest.mock import patch

from src.AI_Banking_Assistant.api.main import app


class StubRetriever:
	def __init__(self):
		self.vector_store = type("VectorStoreState", (), {"total_documents": 3})()


class StubOrchestrator:
	def invoke(self, query, **state):
		return {
			"blocked": False,
			"block_reason": "",
			"final_response": f"Stub response for: {query}",
			"sources": ["handbook.txt"],
			"context": "[Source: handbook.txt]\nSavings accounts earn interest.",
			"output_sanitized": False,
		}


def test_health_endpoint_reports_status():
	app.state.retriever = StubRetriever()

	client = TestClient(app)
	with patch("src.AI_Banking_Assistant.api.routes.health.is_model_loaded", return_value=True):
		response = client.get("/health")

	assert response.status_code == 200
	data = response.json()
	assert data["status"] == "ok"
	assert data["index_size"] == 3
	assert data["model_loaded"] is True


def test_chat_endpoint_returns_stubbed_response():
	app.state.orchestrator = StubOrchestrator()
	client = TestClient(app)

	response = client.post(
		"/chat",
		json={
			"query": "What is a savings account?",
			"history": [],
			"top_k": 5,
			"max_tokens": 512,
			"max_context_length": 2000,
		},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["response"].startswith("Stub response for:")
	assert data["sources"] == ["handbook.txt"]
