"""
Project-wide constants for AI Banking Assistant.
Centralizes magic numbers and default values used across modules.
"""

# ===== Model Defaults =====
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
MAX_TOKENS = 512
DEFAULT_DEVICE = "cpu"

# ===== Quantization =====
QUANTIZATION_BITS = 4
QUANTIZATION_TYPE = "nf4"
COMPUTE_DTYPE = "float16"

# ===== RAG Defaults =====
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5

# ===== Guardrails =====
MAX_INPUT_LENGTH = 2048
MAX_OUTPUT_LENGTH = 4096
PII_PLACEHOLDER_MAP = {
    "PERSON": "[NAME]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "US_SSN": "[SSN]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IBAN_CODE": "[IBAN]",
    "US_BANK_NUMBER": "[ACCOUNT]",
    "IP_ADDRESS": "[IP_ADDRESS]",
}

# ===== LoRA Fine-tuning =====
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4

# ===== API =====
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_RATE_LIMIT = 10  # requests per minute

# ===== File Paths =====
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
DATA_FINETUNE_DIR = "data/finetune"
FAISS_INDEX_DIR = "data/faiss_index"
LOGS_DIR = "logs"
SECURITY_AUDIT_LOG = "logs/security_audit.log"

# ===== Supported File Types =====
SUPPORTED_FILE_EXTENSIONS = [".txt", ".pdf", ".docx", ".json", ".jsonl", ".xlsx"]
