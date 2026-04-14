"""Admin route for maintenance operations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from src.AI_Banking_Assistant.api.schemas import AdminResponse
from src.AI_Banking_Assistant.retrieval.indexer import build_index

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/rebuild-index", response_model=AdminResponse)
def rebuild_index(request: Request) -> AdminResponse:
	project_root = Path(request.app.state.project_root)
	build_index(data_dir=str(project_root / "data" / "raw"), output_dir=str(project_root / "data" / "faiss_index"))
	return AdminResponse(action="rebuild-index", status="ok", message="FAISS index rebuilt")

