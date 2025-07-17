#!/usr/bin/env python3
"""
Data preprocessing script for legal documents.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import LegalDataLoader
from src.data.preprocessor import LegalTextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess legal documents for training")
    parser.add_argument("--dataset_path", type=str, default="dataset", 
                       help="Path to the dataset directory")
    parser.add_argument("--output_path", type=str, default="data/processed",
                       help="Path to save processed data")
    parser.add_argument("--clean_text", action="store_true",
                       help="Apply text cleaning")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize data loader
    logger.info("Loading legal documents...")
    data_loader = LegalDataLoader(args.dataset_path)
    documents = data_loader.load_all_documents()
    
    # Print statistics
    stats = data_loader.get_statistics()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize preprocessor
    preprocessor = LegalTextPreprocessor()
    
    # Process documents if cleaning is enabled
    if args.clean_text:
        logger.info("Cleaning text...")
        for doc in documents:
            doc.judgment = preprocessor.clean_text(doc.judgment)
            doc.summary = preprocessor.clean_text(doc.summary)
    
    # Update data loader with processed documents
    data_loader.documents = documents
    
    # Save processed data
    logger.info(f"Saving processed data to {args.output_path}")
    data_loader.save_processed_data(args.output_path)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
