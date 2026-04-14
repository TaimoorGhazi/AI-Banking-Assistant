"""
Document retriever for AI Banking Assistant.
Embeds queries and searches the FAISS index for relevant documents.
"""

from typing import Dict, List, Optional

import numpy as np

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.exceptions import RAGRetrievalError
from src.AI_Banking_Assistant.retrieval.embeddings import generate_embedding
from src.AI_Banking_Assistant.retrieval.vector_store import VectorStore

logger = get_logger(__name__)

# Global retriever instance
_retriever = None


class DocumentRetriever:
    """Retrieves relevant documents from the FAISS index for a given query."""

    def __init__(self, vector_store: VectorStore = None, top_k: int = None):
        config = Config()
        self.top_k = top_k or config.top_k
        self.vector_store = vector_store or VectorStore()

    def load_index(self, path: str = None):
        """Load an existing FAISS index from disk."""
        try:
            self.vector_store.load(path)
            logger.info(f"Retriever loaded index with {self.vector_store.total_documents} documents")
        except Exception as e:
            raise RAGRetrievalError(f"Failed to load index: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve the most relevant documents for a query.
        
        Args:
            query: User question or search query
            top_k: Number of results to return (default: from config)
            
        Returns:
            List of dicts with 'text', 'score', 'source' keys,
            sorted by relevance (lowest distance = most similar)
        """
        if not query or not query.strip():
            raise RAGRetrievalError("Query cannot be empty")

        if self.vector_store.total_documents == 0:
            logger.warning("No documents indexed, returning empty results")
            return []

        k = top_k or self.top_k

        try:
            query_embedding = generate_embedding(query)
            results = self.vector_store.search(query_embedding, top_k=k)
            logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            raise RAGRetrievalError(f"Retrieval failed: {e}")

    def format_context(self, results: List[Dict], max_length: int = 2000) -> str:
        """Format retrieval results into a context string for the LLM.
        
        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            source = result.get("source", "unknown")
            snippet = f"[Source: {source}]\n{text}"

            if total_length + len(snippet) > max_length:
                break

            context_parts.append(snippet)
            total_length += len(snippet)

        return "\n\n---\n\n".join(context_parts)


def get_retriever(index_path: str = None) -> DocumentRetriever:
    """Get or create a global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
        if index_path:
            _retriever.load_index(index_path)
    return _retriever
