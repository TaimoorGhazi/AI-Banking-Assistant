"""Health route for the SecureBank API."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.AI_Banking_Assistant.api.schemas import HealthResponse
from src.AI_Banking_Assistant.llm.model_loader import is_model_loaded

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
	retriever = request.app.state.retriever
	model_loaded = is_model_loaded()

	return HealthResponse(
		status="ok",
		model_loaded=model_loaded,
		index_size=getattr(retriever.vector_store, "total_documents", 0),
	)

