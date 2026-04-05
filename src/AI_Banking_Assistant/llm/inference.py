"""
LLM inference engine for AI Banking Assistant.
Handles text generation with the loaded model.
"""

from typing import Dict, List, Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.exceptions import ModelInferenceError
from src.AI_Banking_Assistant.llm.model_loader import get_model

logger = get_logger(__name__)


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

    try:
        import torch
        model, tokenizer = get_model()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
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
        raise ModelInferenceError(f"Generation failed: {e}")


def generate_with_context(
    query: str,
    context: str,
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

    full_prompt = format_rag_prompt(query, context, system_prompt)
    return generate_response(full_prompt, max_tokens=max_tokens)


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
        raise ModelInferenceError(f"Chat generation failed: {e}")
