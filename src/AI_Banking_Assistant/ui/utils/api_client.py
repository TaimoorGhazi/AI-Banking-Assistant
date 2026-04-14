"""HTTP client helpers for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ApiClient:
	def __init__(self, base_url: str = "http://localhost:8000"):
		self.base_url = base_url.rstrip("/")

	def health(self) -> Dict[str, Any]:
		response = requests.get(f"{self.base_url}/health", timeout=15)
		response.raise_for_status()
		return response.json()

	def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		response = requests.post(f"{self.base_url}/chat", json=payload, timeout=120)
		response.raise_for_status()
		return response.json()

	def upload_document(self, file_path: str, rebuild_index: bool = True) -> Dict[str, Any]:
		with open(file_path, "rb") as handle:
			files = {"file": (Path(file_path).name, handle)}
			response = requests.post(
				f"{self.base_url}/upload_doc",
				files=files,
				params={"rebuild_index": str(rebuild_index).lower()},
				timeout=300,
			)
		response.raise_for_status()
		return response.json()

