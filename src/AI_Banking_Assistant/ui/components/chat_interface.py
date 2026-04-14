"""Streamlit chat interface component."""

from __future__ import annotations

from typing import List

import streamlit as st

from src.AI_Banking_Assistant.ui.utils.session_state import append_message


def render_chat_interface(messages: List[dict]) -> str | None:
	for message in messages:
		with st.chat_message(message["role"]):
			st.write(message["content"])

	if not messages:
		st.markdown(
			"""
			<div class="center-hero">
				<h3>Ask Anything About Your Banking Services</h3>
				<p>Quick answers, clear guidance, and document-backed responses.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)

		left, center, right = st.columns([1, 2.6, 1])
		prompt = ""
		submitted = False
		with center:
			with st.form("chat_prompt_form", clear_on_submit=True):
				prompt = st.text_input(
					"Ask about SecureBank services",
					placeholder="Type your question here...",
					label_visibility="collapsed",
				)
				submitted = st.form_submit_button("Send", use_container_width=True)

		if submitted and prompt.strip():
			prompt = prompt.strip()
			append_message("user", prompt)
			with st.chat_message("user"):
				st.write(prompt)
			return prompt
		return None

	prompt = st.chat_input("Ask about SecureBank services")
	if prompt and prompt.strip():
		prompt = prompt.strip()
		append_message("user", prompt)
		with st.chat_message("user"):
			st.write(prompt)
		return prompt
	return None

