"""
LLM model loader for AI Banking Assistant.
Loads Llama-3.2-3B-Instruct with 4-bit quantization.
"""

from typing import Optional, Tuple

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import (
    QUANTIZATION_BITS,
    QUANTIZATION_TYPE,
    COMPUTE_DTYPE,
)
from src.AI_Banking_Assistant.core.exceptions import ModelLoadError

logger = get_logger(__name__)

# Cache loaded model and tokenizer
_model = None
_tokenizer = None


def load_model(
    model_name: str = None,
    device: str = None,
    use_quantization: bool = True,
):
    """Load the LLM model with optional 4-bit quantization.
    
    Args:
        model_name: HuggingFace model name (default: from config)
        device: Device to load on (default: from config)
        use_quantization: Whether to apply 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    config = Config()
    model_name = model_name or config.model_name
    device = device or config.device

    logger.info(f"Loading model: {model_name} (device={device}, quantized={use_quantization})")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=config.hf_token,
            trust_remote_code=True,
        )

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Configure quantization
        model_kwargs = {
            "token": config.hf_token,
            "trust_remote_code": True,
        }

        if use_quantization and device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=QUANTIZATION_TYPE,
                bnb_4bit_compute_dtype=getattr(torch, COMPUTE_DTYPE),
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device
            if device != "cpu":
                model_kwargs["torch_dtype"] = torch.float16

        _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        _model.eval()

        param_count = sum(p.numel() for p in _model.parameters())
        logger.info(f"Model loaded: {param_count / 1e6:.1f}M parameters")

        return _model, _tokenizer

    except ImportError as e:
        raise ModelLoadError(f"Missing dependency: {e}. Run: pip install torch transformers bitsandbytes")
    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{model_name}': {e}")


def get_model():
    """Get the cached model, loading it if necessary."""
    global _model, _tokenizer
    if _model is None:
        load_model()
    return _model, _tokenizer


def unload_model():
    """Unload the model from memory."""
    global _model, _tokenizer
    if _model is not None:
        import torch
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded from memory")
