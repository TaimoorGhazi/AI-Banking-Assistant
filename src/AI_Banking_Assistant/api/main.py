"""FastAPI application for SecureBank AI Assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.AI_Banking_Assistant.agents import AgentOrchestrator, RAGAgent, ResponseAgent
from src.AI_Banking_Assistant.api.middleware.cors import add_cors_middleware
from src.AI_Banking_Assistant.api.routes import admin, chat, health, upload
from src.AI_Banking_Assistant.core.config import PROJECT_ROOT
from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.retrieval.retriever import DocumentRetriever

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.project_root = PROJECT_ROOT
	app.state.retriever = DocumentRetriever()
	index_path = Path(PROJECT_ROOT) / "data" / "faiss_index"
	try:
		app.state.retriever.load_index(str(index_path))
	except Exception:
		logger.info("No FAISS index available at startup; continuing with an empty retriever")
	app.state.model_loaded = False
	app.state.orchestrator = AgentOrchestrator(
		rag_agent=RAGAgent(retriever=app.state.retriever),
		response_agent=ResponseAgent(),
	)
	yield


app = FastAPI(title="SecureBank AI Assistant", version="0.1.0", lifespan=lifespan)

add_cors_middleware(app)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)
app.include_router(admin.router)


@app.get("/")
def root() -> dict[str, str]:
	return {"status": "ok", "service": "SecureBank AI Assistant"}


def run() -> None:
	import uvicorn

	uvicorn.run("src.AI_Banking_Assistant.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
	run()

