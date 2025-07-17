"""
Legal RAG (Retrieval-Augmented Generation) system for legal document querying.
"""

import os
import yaml
import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import json

logger = logging.getLogger(__name__)

class LegalRAG:
    """
    RAG system specifically designed for legal document retrieval and generation.
    """

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        self.config = self._load_config(config_path)
        self.embeddings_model = None
        self.vector_store = None
        self.documents = []
        self.document_metadata = []

    def _load_config(self, config_path: str) -> Dict:
        """Load RAG configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_embeddings(self):
        """Initialize the sentence transformer model for embeddings."""
        model_name = self.config['embeddings']['model_name']

        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading embeddings model: {model_name}")
        logger.info(f"Using device: {device}")
        self.embeddings_model = SentenceTransformer(model_name, device=device)
        logger.info("Embeddings model loaded successfully")

    def load_documents(self, documents_path: str):
        """Load legal documents from the dataset."""
        from src.data.data_loader import LegalDataLoader

        logger.info(f"Loading documents from {documents_path}")

        # Load documents using our data loader
        data_loader = LegalDataLoader(documents_path)
        legal_docs = data_loader.load_all_documents()

        # Convert to LangChain documents
        self.documents = []
        self.document_metadata = []

        for doc in legal_docs:
            # Create document chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['documents']['chunk_size'],
                chunk_overlap=self.config['documents']['chunk_overlap'],
                separators=self.config['documents']['separators']
            )

            # Split judgment text
            judgment_chunks = text_splitter.split_text(doc.judgment)

            for i, chunk in enumerate(judgment_chunks):
                # Create LangChain document
                langchain_doc = Document(
                    page_content=chunk,
                    metadata={
                        'doc_id': doc.id,
                        'chunk_id': i,
                        'jurisdiction': doc.jurisdiction,
                        'case_type': doc.case_type,
                        'split': doc.split,
                        'summary': doc.summary,
                        'source': 'judgment'
                    }
                )

                self.documents.append(langchain_doc)
                self.document_metadata.append(langchain_doc.metadata)

            # Also add summary as a separate document
            summary_doc = Document(
                page_content=doc.summary,
                metadata={
                    'doc_id': doc.id,
                    'chunk_id': 0,
                    'jurisdiction': doc.jurisdiction,
                    'case_type': doc.case_type,
                    'split': doc.split,
                    'summary': doc.summary,
                    'source': 'summary'
                }
            )

            self.documents.append(summary_doc)
            self.document_metadata.append(summary_doc.metadata)

        logger.info(f"Loaded {len(self.documents)} document chunks")

    def build_vector_index(self):
        """Build FAISS vector index from documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        if self.embeddings_model is None:
            self.initialize_embeddings()

        logger.info("Building vector index...")

        # Extract text content
        texts = [doc.page_content for doc in self.documents]

        # Generate embeddings in batches
        batch_size = self.config['embeddings']['batch_size']
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings_model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.vector_store.add(embeddings)

        logger.info(f"Vector index built with {self.vector_store.ntotal} vectors")

    def save_index(self, index_path: str):
        """Save the vector index and metadata."""
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.vector_store, str(index_path / "faiss_index.bin"))

        # Save document metadata
        with open(index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.document_metadata, f)

        # Save documents
        with open(index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)

        # Save config
        with open(index_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Index saved to {index_path}")

    def load_index(self, index_path: str):
        """Load a pre-built vector index."""
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index path {index_path} does not exist")

        # Load FAISS index
        self.vector_store = faiss.read_index(str(index_path / "faiss_index.bin"))

        # Load metadata
        with open(index_path / "metadata.pkl", 'rb') as f:
            self.document_metadata = pickle.load(f)

        # Load documents
        with open(index_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)

        # Initialize embeddings model
        if self.embeddings_model is None:
            self.initialize_embeddings()

        logger.info(f"Index loaded from {index_path}")

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents."""
        if self.vector_store is None:
            raise ValueError("Vector index not built. Call build_vector_index() or load_index() first.")

        top_k = top_k or self.config['retrieval']['top_k']

        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.vector_store.search(query_embedding, top_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'score': float(score),
                    'content': self.documents[idx].page_content
                }
                results.append(result)

        # Filter by similarity threshold
        threshold = self.config['retrieval']['similarity_threshold']
        results = [r for r in results if r['score'] >= threshold]

        return results

    def query(self, question: str, llm_model=None) -> Dict:
        """
        Answer a question using RAG approach.
        """
        # Retrieve relevant documents
        relevant_docs = self.search(question)

        if not relevant_docs:
            return {
                'answer': "I couldn't find relevant legal documents to answer your question.",
                'sources': [],
                'confidence': 0.0
            }

        # Prepare context
        context_parts = []
        sources = []

        for doc in relevant_docs:
            context_parts.append(f"Document: {doc['metadata']['doc_id']}")
            context_parts.append(f"Content: {doc['content']}")
            context_parts.append("---")

            sources.append({
                'doc_id': doc['metadata']['doc_id'],
                'jurisdiction': doc['metadata']['jurisdiction'],
                'score': doc['score'],
                'source_type': doc['metadata']['source']
            })

        context = "\n".join(context_parts)

        # Create prompt for LLM
        prompt = f"""Based on the following legal documents, please answer the question. Provide a comprehensive answer citing relevant legal principles and precedents.

Context:
{context}

Question: {question}

Answer:"""

        # If LLM model is provided, generate answer
        if llm_model:
            try:
                answer = llm_model.summarize(prompt)  # Using our summarizer as LLM
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answer = "Error generating answer with the language model."
        else:
            answer = "Please provide an LLM model to generate answers. Currently showing retrieved context only."

        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'confidence': np.mean([doc['score'] for doc in relevant_docs])
        }

    def get_similar_cases(self, case_description: str, top_k: int = 5) -> List[Dict]:
        """Find similar legal cases based on description."""
        results = self.search(case_description, top_k)

        # Group by document ID to avoid duplicate cases
        unique_cases = {}
        for result in results:
            doc_id = result['metadata']['doc_id']
            if doc_id not in unique_cases or result['score'] > unique_cases[doc_id]['score']:
                unique_cases[doc_id] = {
                    'doc_id': doc_id,
                    'jurisdiction': result['metadata']['jurisdiction'],
                    'case_type': result['metadata']['case_type'],
                    'summary': result['metadata']['summary'],
                    'score': result['score'],
                    'relevant_content': result['content']
                }

        # Sort by score and return top cases
        similar_cases = list(unique_cases.values())
        similar_cases.sort(key=lambda x: x['score'], reverse=True)

        return similar_cases[:top_k]

    @classmethod
    def load(cls, index_path: str, config_path: str = "config/rag_config.yaml"):
        """Load a pre-built RAG system."""
        rag = cls(config_path)
        rag.load_index(index_path)
        return rag

def main():
    """Example usage of the Legal RAG system."""
    # Initialize RAG system
    rag = LegalRAG()

    # Load documents
    rag.load_documents("dataset")

    # Build vector index
    rag.build_vector_index()

    # Save index
    rag.save_index("data/rag_index")

    # Example search
    results = rag.search("property tax deduction income tax")
    print(f"Found {len(results)} relevant documents")

    for result in results[:3]:
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:200]}...")
        print("---")

if __name__ == "__main__":
    main()
