"""
Structured logging setup for AI Banking Assistant.
Provides colored console output and rotating file logging.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from colorlog import ColoredFormatter
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_logger(name: str, log_level: str = None, log_file: str = None) -> logging.Logger:
    """Create a configured logger with console and file handlers.
    
    Args:
        name: Logger name (typically module name, e.g., 'preprocessing')
        log_level: Override log level (default: from settings or INFO)
        log_file: Override log file path (default: from settings or logs/app.log)
        
    Returns:
        Configured logger instance
        
    Usage:
        from src.AI_Banking_Assistant.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.error("Something failed", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Determine log level
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # ---- Console Handler (colored) ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if HAS_COLORLOG:
        console_format = ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ---- File Handler (rotating) ----
    log_file_path = log_file or os.getenv("LOG_FILE", "logs/app.log")
    log_path = PROJECT_ROOT / log_file_path

    # Create logs directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Prevent log propagation to root logger
    logger.propagate = False

    return logger
