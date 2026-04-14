"""
Embedding generation using sentence-transformers.
Converts text into dense vector representations for FAISS retrieval.
"""

from typing import List, Optional, Union

import numpy as np

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION
from src.AI_Banking_Assistant.core.exceptions import EmbeddingError

logger = get_logger(__name__)

# Cache the model globally to avoid reloading
_model = None


def load_embedding_model(model_name: str = None):
    """Load the sentence-transformer embedding model.
    
    Args:
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        
    Returns:
        SentenceTransformer model instance
    """
    global _model

    if _model is not None:
        return _model

    model_name = model_name or DEFAULT_EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        return _model
    except ImportError:
        raise EmbeddingError("sentence-transformers not installed. Run: pip install sentence-transformers")
    except Exception as e:
        raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}")


def generate_embedding(text: str, model_name: str = None) -> np.ndarray:
    """Generate embedding vector for a single text.
    
    Args:
        text: Input text to embed
        model_name: Optional model override
        
    Returns:
        Numpy array of shape (384,) for MiniLM
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot generate embedding for empty text")

    model = load_embedding_model(model_name)

    try:
        embedding = model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embedding: {e}")


def batch_encode(texts: List[str], model_name: str = None, batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a batch of texts.
    
    Args:
        texts: List of input texts
        model_name: Optional model override
        batch_size: Encoding batch size
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    if not texts:
        raise EmbeddingError("Cannot encode empty text list")

    # Filter out empty strings
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        raise EmbeddingError("All texts are empty after filtering")

    model = load_embedding_model(model_name)

    try:
        embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=len(valid_texts) > 100,
            normalize_embeddings=True,
        )
        logger.info(f"Encoded {len(valid_texts)} texts -> shape {embeddings.shape}")
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        raise EmbeddingError(f"Batch encoding failed: {e}")


def get_embedding_dimension(model_name: str = None) -> int:
    """Get the output dimension of the embedding model."""
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()
