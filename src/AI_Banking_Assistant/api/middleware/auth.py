"""Placeholder authentication dependency for future secured endpoints."""

from __future__ import annotations

from fastapi import Header, HTTPException, status


def require_authentication(authorization: str | None = Header(default=None)) -> None:
	if authorization is None:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

