#!/usr/bin/env python3
"""
Training script for legal document summarization model.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import yaml

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set MPS memory management for Mac
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.summarizer import LegalSummarizer
from src.utils.device_utils import log_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train legal document summarization model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--train_data", type=str, default="data/processed/train_split.json",
                       help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/processed/validation.json",
                       help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for trained model")

    args = parser.parse_args()

    # Load training configuration
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)

    # Log device information
    log_device_info()

    # Initialize summarizer
    logger.info("Initializing legal summarizer...")
    summarizer = LegalSummarizer(args.model_config)

    # Load base model
    logger.info("Loading base model...")
    summarizer.load_base_model()

    # Setup LoRA
    logger.info("Setting up LoRA configuration...")
    summarizer.setup_lora()

    # Prepare datasets
    logger.info("Preparing training dataset...")
    train_dataset = summarizer.prepare_dataset(args.train_data, "train")

    val_dataset = None
    if Path(args.val_data).exists():
        logger.info("Preparing validation dataset...")
        val_dataset = summarizer.prepare_dataset(args.val_data, "validation")

    # Update output directory in training config
    training_config['training']['output_dir'] = args.output_dir

    # Start training
    logger.info("Starting training...")
    summarizer.train(train_dataset, val_dataset, training_config)

    logger.info("Training completed successfully!")
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
