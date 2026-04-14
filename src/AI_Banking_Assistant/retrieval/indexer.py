"""
Index builder for AI Banking Assistant.
Processes documents and builds the FAISS index from scratch.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.AI_Banking_Assistant.core.logger import get_logger
from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.core.constants import DATA_RAW_DIR, DATA_PROCESSED_DIR, FAISS_INDEX_DIR
from src.AI_Banking_Assistant.preprocessing.data_ingestion import ingest_directory
from src.AI_Banking_Assistant.preprocessing.text_cleaner import clean_document
from src.AI_Banking_Assistant.preprocessing.chunking import chunk_document
from src.AI_Banking_Assistant.retrieval.embeddings import batch_encode
from src.AI_Banking_Assistant.retrieval.vector_store import VectorStore

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def build_index(
    data_dir: str = None,
    output_dir: str = None,
    chunk_method: str = "sentences",
    chunk_size: int = None,
) -> VectorStore:
    """Build a FAISS index from raw documents.
    
    Pipeline: load docs -> clean -> chunk -> embed -> index -> save
    
    Args:
        data_dir: Directory with raw documents (default: data/raw/)
        output_dir: Where to save the index (default: data/faiss_index/)
        chunk_method: Chunking strategy ('sentences', 'characters', 'paragraphs')
        chunk_size: Maximum chunk size in characters
        
    Returns:
        The built VectorStore instance
    """
    config = Config()
    data_dir = data_dir or str(PROJECT_ROOT / DATA_RAW_DIR)
    output_dir = output_dir or str(PROJECT_ROOT / FAISS_INDEX_DIR)
    chunk_size = chunk_size or config.chunk_size

    logger.info(f"Building index from {data_dir}")

    # Step 1: Load documents
    documents = ingest_directory(data_dir)
    if not documents:
        logger.warning("No documents found, creating empty index")
        store = VectorStore()
        store.save(output_dir)
        return store

    logger.info(f"Loaded {len(documents)} documents")

    # Step 2: Clean and chunk
    all_chunks = []
    all_metadata = []

    for doc in documents:
        cleaned = clean_document(doc["content"])
        chunks = chunk_document(cleaned, method=chunk_method, chunk_size=chunk_size)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_id": i,
                "total_chunks": len(chunks),
            })

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

    if not all_chunks:
        logger.warning("No chunks created, creating empty index")
        store = VectorStore()
        store.save(output_dir)
        return store

    # Step 3: Generate embeddings
    embeddings = batch_encode(all_chunks)
    logger.info(f"Generated embeddings: shape {embeddings.shape}")

    # Step 4: Build FAISS index
    store = VectorStore(dimension=embeddings.shape[1])
    store.add_documents(embeddings, all_metadata)

    # Step 5: Save
    store.save(output_dir)
    logger.info(f"Index built and saved to {output_dir} ({store.total_documents} vectors)")

    return store


if __name__ == "__main__":
    print("Building FAISS index from raw documents...")
    store = build_index()
    print(f"Done. Indexed {store.total_documents} document chunks.")
