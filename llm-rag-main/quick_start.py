#!/usr/bin/env python3
"""
Quick start script for Legal LLM Fine-tuning and RAG System.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"🚀 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_requirements():
    """Check if required directories and files exist."""
    logger.info("🔍 Checking requirements...")
    
    required_dirs = ["dataset", "config"]
    required_files = ["requirements.txt", "config/model_config.yaml", "config/rag_config.yaml"]
    
    missing = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(f"Directory: {dir_path}")
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(f"File: {file_path}")
    
    if missing:
        logger.error("❌ Missing required files/directories:")
        for item in missing:
            logger.error(f"  - {item}")
        return False
    
    logger.info("✅ All requirements satisfied")
    return True

def install_dependencies():
    """Install Python dependencies."""
    logger.info("📦 Installing dependencies...")
    
    commands = [
        "pip install -r requirements.txt",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\""
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "data/processed",
        "data/rag_index", 
        "models/fine_tuned",
        "results",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {dir_path}")
    
    return True

def preprocess_data():
    """Preprocess the legal documents."""
    return run_command(
        "python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed --clean_text",
        "Preprocessing legal documents"
    )

def build_rag_index():
    """Build the RAG vector index."""
    return run_command(
        "python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index",
        "Building RAG vector index"
    )

def test_system():
    """Test the system components."""
    logger.info("🧪 Testing system components...")
    
    test_script = """
import sys
sys.path.append('.')

try:
    from src.data.data_loader import LegalDataLoader
    from src.models.summarizer import LegalSummarizer
    from src.rag.legal_rag import LegalRAG
    from src.chatbot.legal_chatbot import LegalChatbot
    
    print("✅ All imports successful")
    
    # Test data loader
    loader = LegalDataLoader("dataset")
    docs = loader.load_all_documents()
    print(f"✅ Data loader: {len(docs)} documents loaded")
    
    # Test RAG system
    try:
        rag = LegalRAG.load("data/rag_index")
        results = rag.search("property tax", top_k=3)
        print(f"✅ RAG system: {len(results)} results found")
    except Exception as e:
        print(f"⚠️  RAG system: {e}")
    
    print("🎉 System test completed!")
    
except Exception as e:
    print(f"❌ System test failed: {e}")
    sys.exit(1)
"""
    
    with open("test_system.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_system.py", "Testing system components")
    
    # Clean up
    if Path("test_system.py").exists():
        Path("test_system.py").unlink()
    
    return success

def main():
    """Main setup function."""
    print("🏛️  Legal LLM Fine-tuning and RAG System - Quick Start")
    print("=" * 60)
    
    steps = [
        ("Check Requirements", check_requirements),
        ("Install Dependencies", install_dependencies),
        ("Setup Directories", setup_directories),
        ("Preprocess Data", preprocess_data),
        ("Build RAG Index", build_rag_index),
        ("Test System", test_system)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 40)
        
        if not step_func():
            logger.error(f"❌ Setup failed at step: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Fine-tune the model (optional):")
    print("   python scripts/train_summarization.py")
    print("\n2. Launch the web application:")
    print("   streamlit run web_app/app.py")
    print("\n3. Or use the Makefile:")
    print("   make webapp")
    print("\n💡 For help: python quick_start.py --help")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Legal LLM Quick Start Script

Usage: python quick_start.py

This script will:
1. Check all requirements
2. Install Python dependencies
3. Set up necessary directories
4. Preprocess legal documents
5. Build RAG vector index
6. Test system components

After completion, you can:
- Launch web app: streamlit run web_app/app.py
- Train model: python scripts/train_summarization.py
- Use Makefile: make webapp

For more information, see README.md
        """)
    else:
        main()
