"""Simple rate-limit helper placeholder for the API layer."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def enforce_api_limit(request: Request, limit_key: str = "rate_limit") -> None:
	limiter = getattr(request.app.state, limit_key, None)
	if limiter is None:
		return

	if getattr(limiter, "blocked", False):
		raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")

