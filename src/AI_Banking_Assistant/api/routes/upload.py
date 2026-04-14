"""Document upload route for the SecureBank API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.AI_Banking_Assistant.api.schemas import UploadResponse
from src.AI_Banking_Assistant.core.constants import DATA_RAW_DIR
from src.AI_Banking_Assistant.retrieval.retriever import DocumentRetriever
from src.AI_Banking_Assistant.retrieval.indexer import build_index
from src.AI_Banking_Assistant.agents import AgentOrchestrator, RAGAgent, ResponseAgent

router = APIRouter(prefix="/upload_doc", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...), rebuild_index: bool = True) -> UploadResponse:
	suffix = Path(file.filename or "").suffix.lower()
	if suffix not in {".txt", ".pdf", ".docx"}:
		raise HTTPException(status_code=400, detail="Unsupported file type")

	raw_dir = Path(request.app.state.project_root) / DATA_RAW_DIR
	raw_dir.mkdir(parents=True, exist_ok=True)
	target = raw_dir / (file.filename or "uploaded_document.txt")

	content = await file.read()
	target.write_bytes(content)

	if rebuild_index:
		index_dir = Path(request.app.state.project_root) / "data" / "faiss_index"
		build_index(data_dir=str(raw_dir), output_dir=str(index_dir))
		retriever = DocumentRetriever()
		try:
			retriever.load_index(str(index_dir))
		except Exception:
			pass
		request.app.state.retriever = retriever
		request.app.state.orchestrator = AgentOrchestrator(
			rag_agent=RAGAgent(retriever=request.app.state.retriever),
			response_agent=ResponseAgent(),
		)

	return UploadResponse(filename=target.name, status="ok", message="Document uploaded successfully")

