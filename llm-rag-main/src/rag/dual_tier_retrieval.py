"""
Dual-Tier Retrieval System for Legal Documents.

This module implements a two-tier retrieval approach:
1. Tier 1: Legal Precedent Retrieval (binding precedents)
2. Tier 2: Similar Case Retrieval (factually similar cases)

Uses contrastive learning and dual encoders for separate retrieval heads.
"""

import logging
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of legal documents."""
    PRECEDENT = "precedent"
    SIMILAR_CASE = "similar_case"
    STATUTE = "statute"
    REGULATION = "regulation"

class CourtLevel(Enum):
    """Court hierarchy levels for precedent binding."""
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"
    TRIBUNAL = "tribunal"

@dataclass
class RetrievalResult:
    """Result from dual-tier retrieval."""
    doc_id: str
    content: str
    score: float
    document_type: DocumentType
    court_level: Optional[CourtLevel]
    jurisdiction: str
    binding_strength: float  # 0-1, higher means more binding
    factual_similarity: float  # 0-1, higher means more factually similar
    metadata: Dict

@dataclass
class DualTierResults:
    """Combined results from both retrieval tiers."""
    precedent_results: List[RetrievalResult]
    similar_case_results: List[RetrievalResult]
    combined_ranking: List[RetrievalResult]
    precedent_score: float
    similarity_score: float
    explanation: str

class ContrastiveLegalEncoder(nn.Module):
    """
    Contrastive learning encoder for legal document embeddings.
    Learns to distinguish between precedent relevance and factual similarity.
    """
    
    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 384, projection_dim: int = 256):
        super().__init__()
        
        self.base_encoder = SentenceTransformer(base_model_name)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        # Dual projection heads
        self.precedent_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Temperature parameters for contrastive learning
        self.precedent_temperature = nn.Parameter(torch.tensor(0.07))
        self.similarity_temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual encoders.
        
        Returns:
            Tuple of (precedent_embeddings, similarity_embeddings)
        """
        # Get base embeddings
        base_embeddings = self.base_encoder.encode(texts, convert_to_tensor=True)
        
        # Project through dual heads
        precedent_embeddings = self.precedent_head(base_embeddings)
        similarity_embeddings = self.similarity_head(base_embeddings)
        
        # L2 normalize
        precedent_embeddings = nn.functional.normalize(precedent_embeddings, p=2, dim=1)
        similarity_embeddings = nn.functional.normalize(similarity_embeddings, p=2, dim=1)
        
        return precedent_embeddings, similarity_embeddings

class DualTierRetrieval:
    """
    Dual-tier retrieval system for legal documents.
    
    Separates legally binding precedents from factually similar cases
    using specialized encoders and retrieval strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the dual-tier retrieval system."""
        self.config = self._load_config(config_path)
        
        # Initialize encoders
        self.contrastive_encoder = None
        self.base_encoder = SentenceTransformer(self.config['base_model'])
        
        # Initialize indices
        self.precedent_index = None
        self.similarity_index = None
        
        # Document storage
        self.documents = {}
        self.document_metadata = {}
        
        # Court hierarchy for binding precedent calculation
        self.court_hierarchy = {
            CourtLevel.SUPREME_COURT: 1.0,
            CourtLevel.HIGH_COURT: 0.8,
            CourtLevel.DISTRICT_COURT: 0.6,
            CourtLevel.TRIBUNAL: 0.4
        }
        
        logger.info("Dual-tier retrieval system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for dual-tier retrieval."""
        default_config = {
            'base_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'projection_dim': 256,
            'precedent_weight': 0.7,
            'similarity_weight': 0.3,
            'top_k_precedents': 5,
            'top_k_similar': 5,
            'binding_threshold': 0.6,
            'similarity_threshold': 0.5
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_documents(self, documents_path: str):
        """Load and process legal documents for dual-tier indexing."""
        logger.info(f"Loading documents from {documents_path}")
        
        # Load documents (assuming JSON format)
        with open(documents_path, 'r') as f:
            raw_documents = json.load(f)
        
        processed_docs = []
        for doc in raw_documents:
            processed_doc = self._process_document(doc)
            if processed_doc:
                processed_docs.append(processed_doc)
                self.documents[processed_doc['doc_id']] = processed_doc
                self.document_metadata[processed_doc['doc_id']] = processed_doc['metadata']
        
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs
    
    def _process_document(self, doc: Dict) -> Optional[Dict]:
        """Process a single document for dual-tier indexing."""
        try:
            # Extract document information
            doc_id = doc.get('doc_id', doc.get('id', ''))
            content = doc.get('content', doc.get('text', ''))
            
            if not doc_id or not content:
                return None
            
            # Determine document type and court level
            doc_type = self._classify_document_type(doc)
            court_level = self._determine_court_level(doc)
            jurisdiction = doc.get('jurisdiction', 'unknown')
            
            # Calculate binding strength
            binding_strength = self._calculate_binding_strength(doc_type, court_level, doc)
            
            return {
                'doc_id': doc_id,
                'content': content,
                'document_type': doc_type,
                'court_level': court_level,
                'jurisdiction': jurisdiction,
                'binding_strength': binding_strength,
                'metadata': doc
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    def _classify_document_type(self, doc: Dict) -> DocumentType:
        """Classify document type based on content and metadata."""
        content = doc.get('content', '').lower()
        source = doc.get('source', '').lower()
        
        # Simple classification rules (can be enhanced with ML)
        if any(keyword in content for keyword in ['precedent', 'binding', 'ratio decidendi']):
            return DocumentType.PRECEDENT
        elif any(keyword in source for keyword in ['supreme court', 'high court']):
            return DocumentType.PRECEDENT
        elif 'statute' in content or 'act' in source:
            return DocumentType.STATUTE
        else:
            return DocumentType.SIMILAR_CASE
    
    def _determine_court_level(self, doc: Dict) -> Optional[CourtLevel]:
        """Determine court level from document metadata."""
        source = doc.get('source', '').lower()
        court = doc.get('court', '').lower()
        
        if 'supreme court' in source or 'supreme court' in court:
            return CourtLevel.SUPREME_COURT
        elif 'high court' in source or 'high court' in court:
            return CourtLevel.HIGH_COURT
        elif 'district court' in source or 'district court' in court:
            return CourtLevel.DISTRICT_COURT
        elif 'tribunal' in source or 'tribunal' in court:
            return CourtLevel.TRIBUNAL
        else:
            return None
    
    def _calculate_binding_strength(self, doc_type: DocumentType, 
                                   court_level: Optional[CourtLevel], doc: Dict) -> float:
        """Calculate binding strength of a document."""
        base_strength = 0.5
        
        # Boost for precedent documents
        if doc_type == DocumentType.PRECEDENT:
            base_strength += 0.3
        
        # Boost based on court hierarchy
        if court_level:
            base_strength += self.court_hierarchy[court_level] * 0.2
        
        # Additional factors
        if 'landmark' in doc.get('content', '').lower():
            base_strength += 0.1
        
        return min(base_strength, 1.0)
    
    def build_dual_indices(self):
        """Build separate indices for precedent and similarity retrieval."""
        logger.info("Building dual-tier indices")
        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        # Prepare documents for encoding
        doc_texts = [doc['content'] for doc in self.documents.values()]
        doc_ids = list(self.documents.keys())
        
        # Generate embeddings using base encoder
        logger.info("Generating base embeddings...")
        base_embeddings = self.base_encoder.encode(doc_texts, show_progress_bar=True)
        
        # If contrastive encoder is available, use dual embeddings
        if self.contrastive_encoder:
            logger.info("Generating contrastive embeddings...")
            precedent_embeddings, similarity_embeddings = self.contrastive_encoder(doc_texts)
            precedent_embeddings = precedent_embeddings.cpu().numpy()
            similarity_embeddings = similarity_embeddings.cpu().numpy()
        else:
            # Use base embeddings for both (can be enhanced later)
            precedent_embeddings = base_embeddings
            similarity_embeddings = base_embeddings
        
        # Build FAISS indices
        logger.info("Building FAISS indices...")
        
        # Precedent index
        self.precedent_index = faiss.IndexFlatIP(precedent_embeddings.shape[1])
        self.precedent_index.add(precedent_embeddings.astype('float32'))
        
        # Similarity index
        self.similarity_index = faiss.IndexFlatIP(similarity_embeddings.shape[1])
        self.similarity_index.add(similarity_embeddings.astype('float32'))
        
        # Store embeddings for later use
        self.precedent_embeddings = precedent_embeddings
        self.similarity_embeddings = similarity_embeddings
        self.doc_ids = doc_ids
        
        logger.info("Dual-tier indices built successfully")
    
    def search(self, query: str, top_k: int = 10) -> DualTierResults:
        """
        Perform dual-tier retrieval search.
        
        Args:
            query: Search query
            top_k: Number of results to return from each tier
            
        Returns:
            DualTierResults: Combined results from both tiers
        """
        logger.debug(f"Performing dual-tier search for: {query[:50]}...")
        
        if not self.precedent_index or not self.similarity_index:
            raise ValueError("Indices not built. Call build_dual_indices() first.")
        
        # Encode query
        query_embedding = self.base_encoder.encode([query])
        
        # Tier 1: Precedent retrieval
        precedent_results = self._search_precedents(query_embedding, top_k)
        
        # Tier 2: Similar case retrieval
        similar_case_results = self._search_similar_cases(query_embedding, top_k)
        
        # Combine and rank results
        combined_results = self._combine_and_rank(precedent_results, similar_case_results)
        
        # Calculate tier scores
        precedent_score = np.mean([r.score for r in precedent_results]) if precedent_results else 0
        similarity_score = np.mean([r.score for r in similar_case_results]) if similar_case_results else 0
        
        # Generate explanation
        explanation = self._generate_search_explanation(precedent_results, similar_case_results)
        
        return DualTierResults(
            precedent_results=precedent_results,
            similar_case_results=similar_case_results,
            combined_ranking=combined_results,
            precedent_score=precedent_score,
            similarity_score=similarity_score,
            explanation=explanation
        )
    
    def _search_precedents(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Search for binding precedents."""
        # Search precedent index
        scores, indices = self.precedent_index.search(query_embedding.astype('float32'), top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # Filter for precedent documents and high binding strength
            if (doc['document_type'] == DocumentType.PRECEDENT or 
                doc['binding_strength'] >= self.config['binding_threshold']):
                
                result = RetrievalResult(
                    doc_id=doc_id,
                    content=doc['content'],
                    score=float(score),
                    document_type=doc['document_type'],
                    court_level=doc['court_level'],
                    jurisdiction=doc['jurisdiction'],
                    binding_strength=doc['binding_strength'],
                    factual_similarity=0.0,  # Will be calculated separately
                    metadata=doc['metadata']
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _search_similar_cases(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """Search for factually similar cases."""
        # Search similarity index
        scores, indices = self.similarity_index.search(query_embedding.astype('float32'), top_k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # Calculate factual similarity score
            factual_similarity = self._calculate_factual_similarity(query_embedding, idx)
            
            # Filter for high factual similarity
            if factual_similarity >= self.config['similarity_threshold']:
                result = RetrievalResult(
                    doc_id=doc_id,
                    content=doc['content'],
                    score=float(score),
                    document_type=doc['document_type'],
                    court_level=doc['court_level'],
                    jurisdiction=doc['jurisdiction'],
                    binding_strength=doc['binding_strength'],
                    factual_similarity=factual_similarity,
                    metadata=doc['metadata']
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _calculate_factual_similarity(self, query_embedding: np.ndarray, doc_idx: int) -> float:
        """Calculate factual similarity between query and document."""
        doc_embedding = self.similarity_embeddings[doc_idx:doc_idx+1]
        similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
        return float(similarity)
    
    def _combine_and_rank(self, precedent_results: List[RetrievalResult], 
                         similar_case_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Combine and rank results from both tiers."""
        all_results = []
        
        # Add precedent results with weighted scores
        for result in precedent_results:
            weighted_score = (result.score * self.config['precedent_weight'] + 
                            result.binding_strength * 0.3)
            result.score = weighted_score
            all_results.append(result)
        
        # Add similar case results with weighted scores
        for result in similar_case_results:
            weighted_score = (result.score * self.config['similarity_weight'] + 
                            result.factual_similarity * 0.3)
            result.score = weighted_score
            all_results.append(result)
        
        # Remove duplicates and sort by score
        seen_docs = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.doc_id not in seen_docs:
                unique_results.append(result)
                seen_docs.add(result.doc_id)
        
        return unique_results[:10]  # Return top 10 combined results
    
    def _generate_search_explanation(self, precedent_results: List[RetrievalResult], 
                                   similar_case_results: List[RetrievalResult]) -> str:
        """Generate explanation for search results."""
        explanation_parts = []
        
        if precedent_results:
            precedent_count = len(precedent_results)
            avg_binding = np.mean([r.binding_strength for r in precedent_results])
            explanation_parts.append(
                f"Found {precedent_count} binding precedents with average binding strength {avg_binding:.2f}"
            )
        
        if similar_case_results:
            similar_count = len(similar_case_results)
            avg_similarity = np.mean([r.factual_similarity for r in similar_case_results])
            explanation_parts.append(
                f"Found {similar_count} factually similar cases with average similarity {avg_similarity:.2f}"
            )
        
        if not explanation_parts:
            return "No relevant precedents or similar cases found."
        
        return ". ".join(explanation_parts) + "."
    
    def save_indices(self, save_path: str):
        """Save the dual-tier indices and metadata."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        if self.precedent_index:
            faiss.write_index(self.precedent_index, str(save_path / "precedent_index.faiss"))
        if self.similarity_index:
            faiss.write_index(self.similarity_index, str(save_path / "similarity_index.faiss"))
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'doc_ids': self.doc_ids,
            'config': self.config
        }
        
        with open(save_path / "dual_tier_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save embeddings
        if hasattr(self, 'precedent_embeddings'):
            np.save(save_path / "precedent_embeddings.npy", self.precedent_embeddings)
        if hasattr(self, 'similarity_embeddings'):
            np.save(save_path / "similarity_embeddings.npy", self.similarity_embeddings)
        
        logger.info(f"Dual-tier indices saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'DualTierRetrieval':
        """Load a saved dual-tier retrieval system."""
        load_path = Path(load_path)
        
        # Load metadata
        with open(load_path / "dual_tier_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls()
        instance.config = metadata['config']
        instance.documents = metadata['documents']
        instance.document_metadata = metadata['document_metadata']
        instance.doc_ids = metadata['doc_ids']
        
        # Load FAISS indices
        if (load_path / "precedent_index.faiss").exists():
            instance.precedent_index = faiss.read_index(str(load_path / "precedent_index.faiss"))
        if (load_path / "similarity_index.faiss").exists():
            instance.similarity_index = faiss.read_index(str(load_path / "similarity_index.faiss"))
        
        # Load embeddings
        if (load_path / "precedent_embeddings.npy").exists():
            instance.precedent_embeddings = np.load(load_path / "precedent_embeddings.npy")
        if (load_path / "similarity_embeddings.npy").exists():
            instance.similarity_embeddings = np.load(load_path / "similarity_embeddings.npy")
        
        # Reinitialize base encoder
        instance.base_encoder = SentenceTransformer(instance.config['base_model'])
        
        logger.info(f"Dual-tier retrieval system loaded from {load_path}")
        return instance

def main():
    """Example usage of dual-tier retrieval system."""
    # Initialize system
    dual_retrieval = DualTierRetrieval()
    
    # Example documents (in practice, load from file)
    sample_docs = [
        {
            'doc_id': 'SC_2020_001',
            'content': 'Supreme Court precedent on property tax deduction under Income Tax Act',
            'source': 'Supreme Court of India',
            'court': 'Supreme Court',
            'jurisdiction': 'India'
        },
        {
            'doc_id': 'HC_2019_045',
            'content': 'High Court case involving similar property tax issues',
            'source': 'Delhi High Court',
            'court': 'High Court',
            'jurisdiction': 'Delhi'
        }
    ]
    
    # Save sample docs to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_docs, f)
        temp_file = f.name
    
    try:
        # Load documents and build indices
        dual_retrieval.load_documents(temp_file)
        dual_retrieval.build_dual_indices()
        
        # Perform search
        query = "property tax deduction income tax"
        results = dual_retrieval.search(query)
        
        # Display results
        print("=" * 60)
        print("DUAL-TIER RETRIEVAL RESULTS")
        print("=" * 60)
        
        print(f"\nQuery: {query}")
        print(f"Explanation: {results.explanation}")
        
        print(f"\nPRECEDENT RESULTS ({len(results.precedent_results)}):")
        for i, result in enumerate(results.precedent_results, 1):
            print(f"{i}. {result.doc_id} (Score: {result.score:.3f}, Binding: {result.binding_strength:.3f})")
        
        print(f"\nSIMILAR CASE RESULTS ({len(results.similar_case_results)}):")
        for i, result in enumerate(results.similar_case_results, 1):
            print(f"{i}. {result.doc_id} (Score: {result.score:.3f}, Similarity: {result.factual_similarity:.3f})")
        
        print(f"\nCOMBINED RANKING ({len(results.combined_ranking)}):")
        for i, result in enumerate(results.combined_ranking, 1):
            print(f"{i}. {result.doc_id} (Final Score: {result.score:.3f})")
    
    finally:
        # Clean up
        import os
        os.unlink(temp_file)

if __name__ == "__main__":
    main()
