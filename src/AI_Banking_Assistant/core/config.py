"""
Configuration management for AI Banking Assistant.
Loads settings from config/settings.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Config:
    """Centralized configuration manager.
    
    Loads settings from:
    1. config/settings.yaml (default values)
    2. Environment variables (.env file overrides)
    
    Usage:
        config = Config()
        model_name = config.get("model", "name")
    """

    _instance: Optional["Config"] = None
    _settings: dict = {}

    def __new__(cls) -> "Config":
        """Singleton pattern - only one Config instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self) -> None:
        """Load settings from YAML config file."""
        config_path = PROJECT_ROOT / "config" / "settings.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                self._settings = yaml.safe_load(f) or {}
        else:
            self._settings = {}

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot-style keys.
        
        Args:
            *keys: Nested keys to traverse (e.g., "model", "name")
            default: Default value if key not found
            
        Returns:
            Config value or default
            
        Example:
            config.get("model", "name")  # Returns "meta-llama/Llama-3.2-3B-Instruct"
            config.get("model", "temperature", default=0.7)
        """
        value = self._settings
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def model_name(self) -> str:
        return os.getenv("MODEL_NAME", self.get("model", "name", default="meta-llama/Llama-3.2-3B-Instruct"))

    @property
    def device(self) -> str:
        return os.getenv("DEVICE", self.get("model", "device", default="cpu"))

    @property
    def max_tokens(self) -> int:
        return int(os.getenv("MAX_TOKENS", self.get("model", "max_tokens", default=512)))

    @property
    def temperature(self) -> float:
        return float(os.getenv("TEMPERATURE", self.get("model", "temperature", default=0.7)))

    @property
    def embedding_model(self) -> str:
        return self.get("rag", "embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")

    @property
    def chunk_size(self) -> int:
        return int(self.get("rag", "chunk_size", default=512))

    @property
    def chunk_overlap(self) -> int:
        return int(self.get("rag", "chunk_overlap", default=50))

    @property
    def top_k(self) -> int:
        return int(self.get("rag", "top_k", default=5))

    @property
    def hf_token(self) -> Optional[str]:
        return os.getenv("HF_TOKEN")

    @property
    def groq_api_key(self) -> Optional[str]:
        return os.getenv("GROQ_API_KEY")

    @property
    def mlflow_tracking_uri(self) -> Optional[str]:
        return os.getenv("MLFLOW_TRACKING_URI")

    @property
    def dagshub_user(self) -> Optional[str]:
        return os.getenv("DAGSHUB_USER")

    @property
    def api_host(self) -> str:
        return os.getenv("API_HOST", self.get("api", "host", default="0.0.0.0"))

    @property
    def api_port(self) -> int:
        return int(os.getenv("API_PORT", self.get("api", "port", default=8000)))

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", self.get("logging", "level", default="INFO"))

    @property
    def log_file(self) -> str:
        return os.getenv("LOG_FILE", self.get("logging", "log_file", default="logs/app.log"))

    def __repr__(self) -> str:
        return f"Config(model={self.model_name}, device={self.device})"
