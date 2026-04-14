"""
FAISS vector store management for AI Banking Assistant.
Handles index creation, document storage, save/load, and search.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.constants import EMBEDDING_DIMENSION, FAISS_INDEX_DIR
from src.AI_Banking_Assistant.core.exceptions import IndexError

logger = get_logger(__name__)

# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class VectorStore:
    """FAISS-based vector store for document embeddings.
    
    Stores document embeddings in a FAISS index along with
    metadata (original text, source file, chunk id) for retrieval.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict] = []
        self._init_index()

    def _init_index(self):
        """Initialize an empty FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Initialized FAISS IndexFlatL2 (dim={self.dimension})")
        except ImportError:
            raise IndexError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> int:
        """Add document embeddings to the index.
        
        Args:
            embeddings: Array of shape (n_docs, dimension)
            metadata: List of dicts with keys like 'text', 'source', 'chunk_id'
            
        Returns:
            Number of documents added
        """
        if embeddings.shape[1] != self.dimension:
            raise IndexError(
                f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}"
            )

        if len(embeddings) != len(metadata):
            raise IndexError(
                f"Embeddings and metadata count mismatch: {len(embeddings)} vs {len(metadata)}"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(embeddings)} documents. Total: {self.index.ntotal}")
        return len(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', 'source', etc.
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []

        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = {**self.metadata[idx]}
            result["score"] = float(dist)
            results.append(result)

        logger.info(f"Search returned {len(results)} results")
        return results

    def save(self, path: str = None):
        """Save the FAISS index and metadata to disk.
        
        Args:
            path: Directory to save to (default: data/faiss_index/)
        """
        import faiss

        save_dir = Path(path) if path else PROJECT_ROOT / FAISS_INDEX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        meta_path = save_dir / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index ({self.index.ntotal} vectors) to {save_dir}")

    def load(self, path: str = None):
        """Load a FAISS index and metadata from disk.
        
        Args:
            path: Directory to load from (default: data/faiss_index/)
        """
        import faiss

        load_dir = Path(path) if path else PROJECT_ROOT / FAISS_INDEX_DIR

        index_path = load_dir / "index.faiss"
        meta_path = load_dir / "metadata.pkl"

        if not index_path.exists():
            raise IndexError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        logger.info(f"Loaded index ({self.index.ntotal} vectors) from {load_dir}")

    @property
    def total_documents(self) -> int:
        """Return the total number of indexed documents."""
        return self.index.ntotal if self.index else 0

    def clear(self):
        """Clear the index and metadata."""
        self._init_index()
        self.metadata = []
        logger.info("Cleared vector store")
