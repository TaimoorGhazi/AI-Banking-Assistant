"""Request models for the SecureBank API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
	role: str = Field(pattern="^(system|user|assistant)$")
	content: str = Field(min_length=1)


class ChatRequest(BaseModel):
	query: str = Field(min_length=1, max_length=2048)
	history: List[ChatMessage] = Field(default_factory=list)
	top_k: int = Field(default=5, ge=1, le=20)
	max_tokens: int = Field(default=512, ge=1, le=4096)
	max_context_length: int = Field(default=2000, ge=1, le=8000)
	system_prompt: Optional[str] = None


class UploadRequest(BaseModel):
	filename: Optional[str] = None
	rebuild_index: bool = True

