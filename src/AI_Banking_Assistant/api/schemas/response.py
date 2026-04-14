"""Response models for the SecureBank API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
	source: str
	text: str
	score: Optional[float] = None


class ChatResponse(BaseModel):
	response: str
	blocked: bool = False
	block_reason: Optional[str] = None
	sources: List[str] = Field(default_factory=list)
	retrieved_context: Optional[str] = None
	sanitized: bool = False


class HealthResponse(BaseModel):
	status: str
	model_loaded: bool
	index_size: int


class UploadResponse(BaseModel):
	filename: str
	status: str
	message: str


class AdminResponse(BaseModel):
	action: str
	status: str
	message: str

