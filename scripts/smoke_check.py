"""Minimal smoke check for local API health + chat behavior."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_smoke(base_url: str, timeout: int) -> int:
    health = _get_json(f"{base_url}/health", timeout)
    chat_payload = {
        "query": "What is a savings account?",
        "history": [],
        "top_k": 3,
        "max_tokens": 128,
        "max_context_length": 1000,
    }
    chat = _post_json(f"{base_url}/chat", chat_payload, timeout)

    status_ok = health.get("status") == "ok"
    has_model_flag = isinstance(health.get("model_loaded"), bool)
    has_response_text = bool((chat.get("response") or "").strip())

    print("health.status:", health.get("status"))
    print("health.model_loaded:", health.get("model_loaded"))
    print("chat.blocked:", chat.get("blocked"))
    print("chat.response_length:", len((chat.get("response") or "").strip()))

    if status_ok and has_model_flag and has_response_text:
        print("SMOKE_CHECK=PASS")
        return 0

    print("SMOKE_CHECK=FAIL")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run API health/chat smoke check")
    parser.add_argument("--base-url", default="http://127.0.0.1:8055", help="API base URL")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    args = parser.parse_args()

    try:
        return run_smoke(args.base_url.rstrip("/"), args.timeout)
    except Exception as exc:  # pragma: no cover
        print(f"SMOKE_CHECK=ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
