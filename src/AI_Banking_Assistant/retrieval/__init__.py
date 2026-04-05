"""Retrieval module for AI Banking Assistant.

Provides document embedding, FAISS vector storage,
document retrieval, and index building.
"""

from src.AI_Banking_Assistant.retrieval.embeddings import (
    generate_embedding,
    batch_encode,
    load_embedding_model,
)
from src.AI_Banking_Assistant.retrieval.vector_store import VectorStore
from src.AI_Banking_Assistant.retrieval.retriever import (
    DocumentRetriever,
    get_retriever,
)
from src.AI_Banking_Assistant.retrieval.indexer import build_index
