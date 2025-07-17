# Makefile for Legal LLM Fine-tuning and RAG System

.PHONY: help install setup preprocess train build-rag webapp clean test lint

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  setup        - Complete setup (install + download models)"
	@echo "  preprocess   - Preprocess legal documents"
	@echo "  train        - Train the summarization model"
	@echo "  build-rag    - Build RAG vector index"
	@echo "  webapp       - Launch web application"
	@echo "  clean        - Clean generated files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run code linting"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

setup: install
	@echo "Setting up Legal LLM system..."
	mkdir -p data/processed data/rag_index models results
	@echo "Setup complete!"

preprocess:
	python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed --clean_text

train:
	python scripts/train_summarization.py --train_data data/processed/train_split.json --val_data data/processed/validation.json

build-rag:
	python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index

webapp:
	streamlit run web_app/app.py

clean:
	rm -rf data/processed/* data/rag_index/* results/* models/fine_tuned/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ scripts/ web_app/
	black --check src/ scripts/ web_app/

# Quick start commands
quick-start: setup preprocess build-rag webapp

# Development commands
dev-install: install
	pip install -e .

# Docker commands (if needed)
docker-build:
	docker build -t legal-llm-system .

docker-run:
	docker run -p 8501:8501 legal-llm-system
