"""
LLM inference engine for AI Banking Assistant.
Handles text generation with the loaded model.
"""

import requests
from typing import Dict, List, Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.exceptions import ModelInferenceError
from src.AI_Banking_Assistant.llm.model_loader import get_model

logger = get_logger(__name__)


def _fallback_response(prompt: str) -> str:
    """Return a lightweight local response when full model inference is unavailable."""
    preview = "your request"

    for marker in ("Customer Question:", "Question:", "Customer:", "User:"):
        if marker in prompt:
            preview = prompt.split(marker, 1)[1].strip().splitlines()[0].strip()
            break

    if preview == "your request":
        lines = [line.strip() for line in prompt.splitlines() if line.strip() and line.strip() != "Answer:"]
        if lines:
            preview = lines[-1]

    return (
        "I’m running in local fallback mode on this machine, so I can’t load the full "
        "fine-tuned model right now. I still received your message: "
        f"{preview}"
    )


def generate_response(
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from the LLM given a prompt.
    
    Args:
        prompt: Full prompt string (with system instructions and context)
        max_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling vs greedy decoding
        
    Returns:
        Generated text response
    """
    config = Config()
    max_tokens = max_tokens or config.max_tokens
    temperature = temperature or config.temperature

    if config.use_groq and config.groq_api_key:
        try:
            payload = {
                "model": config.groq_model,
                "messages": [
                    {"role": "system", "content": "You are a banking assistant. Follow provided context and be concise."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                logger.info("Generated response via Groq model: %s", config.groq_model)
                return text
        except Exception as e:
            logger.warning("Groq generation failed; falling back to local model: %s", e)

    try:
        import torch
        model, tokenizer = get_model()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                max_time=config.generation_max_time,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (skip the input)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        logger.info(f"Generated {len(response)} chars (max_tokens={max_tokens}, temp={temperature})")
        return response.strip()

    except Exception as e:
        logger.warning("Generation failed; using fallback response: %s", e)
        return _fallback_response(prompt)


def generate_with_context(
    query: str,
    context: str,
    history: str = "",
    system_prompt: str = None,
    max_tokens: int = None,
) -> str:
    """Generate a response using RAG context.
    
    Args:
        query: User's question
        context: Retrieved document context
        system_prompt: Optional system instruction override
        max_tokens: Maximum new tokens
        
    Returns:
        Generated response
    """
    from src.AI_Banking_Assistant.llm.prompt_templates import format_rag_prompt

    full_prompt = format_rag_prompt(query, context, system_prompt, history=history)
    # Keep RAG answers grounded and stable for factual banking queries.
    return generate_response(
        full_prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
    )


def generate_chat_response(
    messages: List[Dict[str, str]],
    max_tokens: int = None,
) -> str:
    """Generate a response from a list of chat messages.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
                  e.g. [{"role": "user", "content": "Hello"}]
        max_tokens: Maximum new tokens
        
    Returns:
        Generated assistant response
    """
    prompt = ""
    try:
        _, tokenizer = get_model()

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

        return generate_response(prompt, max_tokens=max_tokens)

    except Exception as e:
        logger.warning("Chat generation failed; using fallback response: %s", e)
        if not prompt:
            prompt = " ".join(msg.get("content", "") for msg in messages)
        return _fallback_response(prompt)
