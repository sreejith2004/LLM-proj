#!/usr/bin/env python3
"""
Script to build RAG vector index from legal documents.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.legal_rag import LegalRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index for legal documents")
    parser.add_argument("--documents_path", type=str, default="dataset",
                       help="Path to legal documents dataset")
    parser.add_argument("--index_path", type=str, default="data/rag_index",
                       help="Path to save the vector index")
    parser.add_argument("--config", type=str, default="config/rag_config.yaml",
                       help="Path to RAG configuration file")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = LegalRAG(args.config)
    
    # Initialize embeddings model
    logger.info("Loading embeddings model...")
    rag.initialize_embeddings()
    
    # Load documents
    logger.info(f"Loading documents from {args.documents_path}...")
    rag.load_documents(args.documents_path)
    
    # Build vector index
    logger.info("Building vector index...")
    rag.build_vector_index()
    
    # Save index
    logger.info(f"Saving index to {args.index_path}...")
    rag.save_index(args.index_path)
    
    # Test the index
    logger.info("Testing the index...")
    test_queries = [
        "property tax deduction",
        "income tax appeal",
        "municipal tax liability",
        "court judgment analysis"
    ]
    
    for query in test_queries:
        results = rag.search(query, top_k=3)
        logger.info(f"Query: '{query}' - Found {len(results)} results")
        if results:
            logger.info(f"  Top result score: {results[0]['score']:.3f}")
    
    logger.info("RAG index building completed successfully!")

if __name__ == "__main__":
    main()
