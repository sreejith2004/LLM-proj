"""
Setup script for Legal LLM Fine-tuning and RAG system.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legal-llm-finetune-rag",
    version="1.0.0",
    author="Legal AI Team",
    author_email="legal-ai@example.com",
    description="LLM-Based Court Case Judgment Prediction System with Fine-tuning and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/legal-llm-finetune-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch[cuda]",
            "faiss-gpu",
        ],
    },
    entry_points={
        "console_scripts": [
            "legal-preprocess=scripts.preprocess_data:main",
            "legal-train=scripts.train_summarization:main",
            "legal-build-rag=scripts.build_rag_index:main",
            "legal-webapp=web_app.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)
