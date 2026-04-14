"""Chat route for the SecureBank API."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.AI_Banking_Assistant.api.schemas import ChatRequest, ChatResponse, format_history

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
	orchestrator = request.app.state.orchestrator
	state = orchestrator.invoke(
		payload.query,
		top_k=payload.top_k,
		max_tokens=payload.max_tokens,
		max_context_length=payload.max_context_length,
		system_prompt=payload.system_prompt,
		history=format_history(payload.history),
	)

	return ChatResponse(
		response=state.get("final_response", ""),
		blocked=bool(state.get("blocked", False)),
		block_reason=state.get("block_reason"),
		sources=state.get("sources", []),
		retrieved_context=state.get("context"),
		sanitized=bool(state.get("output_sanitized", False)),
	)

