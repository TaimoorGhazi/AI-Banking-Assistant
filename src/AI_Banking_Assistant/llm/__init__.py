"""LLM module for AI Banking Assistant.

Provides model loading, inference, prompt templates,
and fine-tuning capabilities.
"""

from src.AI_Banking_Assistant.llm.model_loader import load_model, get_model, unload_model
from src.AI_Banking_Assistant.llm.inference import (
    generate_response,
    generate_with_context,
    generate_chat_response,
)
from src.AI_Banking_Assistant.llm.prompt_templates import (
    format_rag_prompt,
    format_chat_prompt,
    build_messages,
    BANKING_SYSTEM_PROMPT,
)
