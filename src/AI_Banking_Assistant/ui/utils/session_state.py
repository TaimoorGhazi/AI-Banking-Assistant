"""Streamlit session state helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st


def initialize_session_state() -> None:
	defaults = {
		"messages": [],
		"api_url": os.getenv("API_URL", "http://localhost:8000"),
		"status_message": "",
		"last_response": "",
	}

	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


def append_message(role: str, content: str) -> None:
	st.session_state.messages.append({"role": role, "content": content})


def clear_messages() -> None:
	st.session_state.messages = []

