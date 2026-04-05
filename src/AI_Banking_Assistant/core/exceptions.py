"""
Custom exception classes for AI Banking Assistant.
Each module raises specific exceptions for clear error handling.
"""


class BankingAssistantError(Exception):
    """Base exception for all Banking Assistant errors."""
    pass


# ===== Configuration Errors =====
class ConfigurationError(BankingAssistantError):
    """Raised when configuration loading or validation fails."""
    pass


# ===== Model Errors =====
class ModelLoadError(BankingAssistantError):
    """Raised when the LLM model fails to load."""
    pass


class ModelInferenceError(BankingAssistantError):
    """Raised when model inference (generation) fails."""
    pass


# ===== RAG Errors =====
class RAGRetrievalError(BankingAssistantError):
    """Raised when document retrieval from FAISS fails."""
    pass


class EmbeddingError(BankingAssistantError):
    """Raised when text embedding generation fails."""
    pass


class IndexError(BankingAssistantError):
    """Raised when FAISS index operations fail."""
    pass


# ===== Preprocessing Errors =====
class DataIngestionError(BankingAssistantError):
    """Raised when document loading/parsing fails."""
    pass


class PIIAnonymizationError(BankingAssistantError):
    """Raised when PII detection or anonymization fails."""
    pass


# ===== Guardrail Errors =====
class GuardrailViolation(BankingAssistantError):
    """Raised when user input violates safety guardrails."""

    def __init__(self, message: str, violation_type: str = "unknown"):
        super().__init__(message)
        self.violation_type = violation_type


class JailbreakDetected(GuardrailViolation):
    """Raised when a jailbreak attempt is detected."""

    def __init__(self, message: str = "Jailbreak attempt detected"):
        super().__init__(message, violation_type="jailbreak")


class PromptInjectionDetected(GuardrailViolation):
    """Raised when prompt injection is detected."""

    def __init__(self, message: str = "Prompt injection detected"):
        super().__init__(message, violation_type="prompt_injection")


# ===== API Errors =====
class RateLimitExceeded(BankingAssistantError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(BankingAssistantError):
    """Raised when authentication fails."""
    pass
