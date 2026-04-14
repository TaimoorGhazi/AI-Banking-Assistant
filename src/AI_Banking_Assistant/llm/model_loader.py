"""
LLM model loader for AI Banking Assistant.
Loads the base model and optionally attaches a local LoRA adapter.
"""

import json
import os
from pathlib import Path
from typing import Optional

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import (
    QUANTIZATION_BITS,
    QUANTIZATION_TYPE,
    COMPUTE_DTYPE,
)
from src.AI_Banking_Assistant.core.exceptions import ModelLoadError
from src.AI_Banking_Assistant.core.config import PROJECT_ROOT

logger = get_logger(__name__)

# Cache loaded model and tokenizer
_model = None
_tokenizer = None


def load_model(
    model_name: str = None,
    device: str = None,
    use_quantization: bool | None = None,
    load_adapter: bool | None = None,
    adapter_path: str = None,
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
    profile = config.runtime_profile
    runtime_mode = config.runtime_mode
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "600")

    model_name = model_name or profile.get("model_name") or config.model_name
    device = device or profile.get("device") or config.device
    if use_quantization is None:
        use_quantization = bool(profile.get("use_quantization", True))
    if load_adapter is None:
        load_adapter = bool(profile.get("load_adapter", True))

    adapter_dir = Path(adapter_path) if adapter_path else Path(PROJECT_ROOT) / "lora_adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"

    if runtime_mode == "gpu_adapter" and str(device).startswith("cpu"):
        logger.warning("gpu_adapter profile requested on CPU device; falling back to cpu_local behavior")
        load_adapter = False

    if load_adapter and adapter_config_path.exists() and device != "cpu":
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            model_name = adapter_config.get("base_model_name_or_path") or model_name or config.model_name
        except Exception:
            model_name = model_name or config.model_name
    else:
        model_name = model_name or config.model_name
        if load_adapter and adapter_config_path.exists() and device == "cpu":
            logger.warning(
                "CPU runtime detected; using fallback model %s instead of the quantized local adapter base",
                model_name,
            )
            load_adapter = False

    logger.info(
        "Loading model: %s (device=%s, quantized=%s, adapter=%s)",
        model_name,
        device,
        use_quantization,
        load_adapter and adapter_config_path.exists(),
    )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=config.hf_token,
            trust_remote_code=True,
            cache_dir=config.hf_cache_dir,
            local_files_only=config.hf_local_files_only,
        )

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Configure quantization
        model_kwargs = {
            "token": config.hf_token,
            "trust_remote_code": True,
            "cache_dir": config.hf_cache_dir,
            "local_files_only": config.hf_local_files_only,
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
        elif device != "cpu":
            model_kwargs["device_map"] = device
            model_kwargs["torch_dtype"] = torch.float16

        _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if load_adapter and adapter_config_path.exists():
            _model = PeftModel.from_pretrained(_model, str(adapter_dir), is_trainable=False)
        if device == "cpu" and hasattr(_model, "to"):
            _model = _model.to("cpu")
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


def is_model_loaded() -> bool:
    """Return whether the model cache currently holds a loaded model."""
    return _model is not None and _tokenizer is not None


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
