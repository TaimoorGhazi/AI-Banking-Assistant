"""Command line wrapper for building the FAISS index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.retrieval.indexer import build_index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the index builder."""
    parser = argparse.ArgumentParser(description="Build the FAISS index from raw documents.")
    parser.add_argument("--data-dir", default=None, help="Directory containing raw documents")
    parser.add_argument("--output-dir", default=None, help="Directory for the generated FAISS index")
    parser.add_argument("--chunk-method", default="sentences", choices=["sentences", "characters", "paragraphs"], help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override the configured chunk size")
    return parser.parse_args()


def main() -> None:
    """Run the index build with the repository defaults."""
    args = parse_args()
    config = Config()

    store = build_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size or config.chunk_size,
    )

    print(f"Built FAISS index with {store.total_documents} vectors.")


if __name__ == "__main__":
    main()
