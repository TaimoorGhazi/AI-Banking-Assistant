"""Streamlit upload widget component."""

from __future__ import annotations

import streamlit as st


def render_upload_widget() -> str | None:
	uploaded = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
	if uploaded is None:
		return None

	temp_path = st.session_state.get("uploaded_temp_path")
	if temp_path is None or st.button("Use this uploaded file"):
		temp_path = f"{uploaded.name}"
		with open(temp_path, "wb") as handle:
			handle.write(uploaded.getbuffer())
		st.session_state["uploaded_temp_path"] = temp_path
		st.sidebar.success(f"Prepared {uploaded.name} for upload")

	return temp_path

