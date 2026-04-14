"""Streamlit application for SecureBank AI Assistant."""

from __future__ import annotations

import streamlit as st

from src.AI_Banking_Assistant.ui.components.chat_interface import render_chat_interface
from src.AI_Banking_Assistant.ui.components.upload_widget import render_upload_widget
from src.AI_Banking_Assistant.ui.utils.api_client import ApiClient
from src.AI_Banking_Assistant.ui.utils.session_state import append_message, clear_messages, initialize_session_state


def apply_theme() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

		html, body, [class*="css"] {
			font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
		}

		.stApp {
			background: radial-gradient(circle at 18% 0%, #f1fff7 0%, #edf8f1 45%, #f8fcf9 100%);
		}

		.main .block-container {
			max-width: 1150px;
			padding-top: 1.1rem;
		}

		section[data-testid="stSidebar"] {
			background: linear-gradient(180deg, #f4fff9 0%, #e8f4ed 100%);
			border-right: 1px solid #d0e6d8;
		}

		h1 {
			color: #143d2f;
			font-weight: 700;
			letter-spacing: 0.2px;
		}

		.stButton > button {
			background: #1f7a53;
			color: #ffffff;
			border: none;
			border-radius: 10px;
		}

		.stButton > button:hover {
			background: #196646;
			color: #ffffff;
		}

		[data-testid="stChatMessage"] {
			background: #ffffffd9;
			border: 1px solid #d6e8dc;
			border-radius: 14px;
			padding: 0.45rem 0.8rem;
			max-width: 100%;
			overflow: hidden;
		}

		[data-testid="stChatMessage"] * {
			overflow-wrap: anywhere;
			word-break: break-word;
		}

		/* Keep bottom chat composer compact with no large background panel. */
		[data-testid="stBottomBlockContainer"] {
			background: transparent !important;
			border: none !important;
			box-shadow: none !important;
			padding: 0 !important;
		}

		[data-testid="stBottomBlockContainer"] > div {
			background: transparent !important;
			border: none !important;
			box-shadow: none !important;
			padding: 0 !important;
		}

		[data-testid="stChatFloatingInputContainer"] {
			background: transparent !important;
			border: none !important;
			box-shadow: none !important;
			padding-top: 0.35rem !important;
			padding-bottom: 0.8rem !important;
		}

		[data-testid="stChatFloatingInputContainer"] > div {
			background: transparent !important;
			border: none !important;
			box-shadow: none !important;
			max-width: 1150px;
			margin: 0 auto;
		}

		[data-testid="stChatInput"] {
			background: transparent !important;
		}

		[data-testid="stChatInput"] > div {
			background: #ffffffeb !important;
			border: 1px solid #c8e0d0 !important;
			border-radius: 14px !important;
			box-shadow: 0 8px 20px rgba(27, 89, 62, 0.08);
		}

		.center-hero {
			text-align: center;
			margin-top: 8vh;
			margin-bottom: 0.8rem;
			color: #275542;
		}

		.center-hero h3 {
			margin-bottom: 0.1rem;
			font-weight: 600;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title="SecureBank AI Assistant", layout="wide")
	initialize_session_state()
	apply_theme()

	client = ApiClient(st.session_state.api_url)
	upload_path = render_upload_widget()

	st.title("SecureBank AI Assistant")

	if st.button("Clear conversation"):
		clear_messages()
		st.rerun()

	prompt = render_chat_interface(st.session_state.messages)

	if prompt:
		is_first_exchange = len(st.session_state.messages) == 1
		payload = {
			"query": prompt,
			"history": st.session_state.messages[:-1],
			"top_k": 5,
			"max_tokens": 512,
			"max_context_length": 2000,
		}

		try:
			result = client.chat(payload)
			response = result["response"]
			append_message("assistant", response)
			st.session_state.last_response = response
			with st.chat_message("assistant"):
				st.write(response)
			if is_first_exchange:
				st.rerun()
		except Exception as exc:
			st.error(f"Failed to get response: {exc}")

	if upload_path:
		try:
			upload_result = client.upload_document(upload_path)
			st.success(upload_result["message"])
		except Exception as exc:
			st.error(f"Upload failed: {exc}")


if __name__ == "__main__":
	main()

