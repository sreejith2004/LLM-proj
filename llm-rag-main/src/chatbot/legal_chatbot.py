"""
AI-powered legal chatbot combining fine-tuned LLM with RAG system.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LegalChatbot:
    """
    Intelligent legal chatbot that combines:
    1. Fine-tuned LLM for legal text generation
    2. RAG system for relevant document retrieval
    3. Conversation memory and context management
    """
    
    def __init__(self, 
                 summarizer_model_path: Optional[str] = None,
                 rag_index_path: Optional[str] = None,
                 conversation_history_limit: int = 10):
        
        self.summarizer = None
        self.rag_system = None
        self.conversation_history = []
        self.conversation_history_limit = conversation_history_limit
        
        # Load models if paths provided
        if summarizer_model_path:
            self.load_summarizer(summarizer_model_path)
        
        if rag_index_path:
            self.load_rag_system(rag_index_path)
    
    def load_summarizer(self, model_path: str):
        """Load the fine-tuned legal summarizer."""
        from src.models.summarizer import LegalSummarizer
        
        try:
            self.summarizer = LegalSummarizer.from_pretrained(model_path)
            logger.info(f"Legal summarizer loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}")
            # Fallback to base model
            self.summarizer = LegalSummarizer()
            self.summarizer.load_base_model()
            logger.info("Loaded base model as fallback")
    
    def load_rag_system(self, index_path: str):
        """Load the RAG system."""
        from src.rag.legal_rag import LegalRAG
        
        try:
            self.rag_system = LegalRAG.load(index_path)
            logger.info(f"RAG system loaded from {index_path}")
        except Exception as e:
            logger.error(f"Failed to load RAG system: {e}")
    
    def chat(self, user_input: str, include_sources: bool = True) -> Dict:
        """
        Main chat interface that handles user queries.
        """
        # Add user message to history
        self._add_to_history("user", user_input)
        
        # Determine query type and route accordingly
        query_type = self._classify_query(user_input)
        
        if query_type == "summarization":
            response = self._handle_summarization_query(user_input)
        elif query_type == "legal_search":
            response = self._handle_legal_search_query(user_input, include_sources)
        elif query_type == "case_analysis":
            response = self._handle_case_analysis_query(user_input, include_sources)
        else:
            response = self._handle_general_query(user_input, include_sources)
        
        # Add bot response to history
        self._add_to_history("assistant", response['answer'])
        
        return response
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query to route to appropriate handler.
        """
        query_lower = query.lower()
        
        # Summarization queries
        if any(keyword in query_lower for keyword in 
               ['summarize', 'summary', 'brief', 'outline', 'key points']):
            return "summarization"
        
        # Legal search queries
        elif any(keyword in query_lower for keyword in 
                ['find cases', 'precedent', 'similar case', 'legal authority', 'case law']):
            return "legal_search"
        
        # Case analysis queries
        elif any(keyword in query_lower for keyword in 
                ['analyze', 'analysis', 'opinion', 'interpretation', 'meaning']):
            return "case_analysis"
        
        # General legal queries
        else:
            return "general"
    
    def _handle_summarization_query(self, query: str) -> Dict:
        """Handle document summarization requests."""
        if self.summarizer is None:
            return {
                'answer': "I don't have access to a summarization model. Please provide a legal document and I'll try to help with the available information.",
                'type': 'error',
                'sources': []
            }
        
        # Extract document text from query if provided
        # This is a simplified approach - in practice, you'd want better document extraction
        if "document:" in query.lower():
            document_text = query.split("document:")[-1].strip()
            
            try:
                summary = self.summarizer.summarize(document_text)
                return {
                    'answer': f"**Legal Document Summary:**\n\n{summary}",
                    'type': 'summarization',
                    'sources': []
                }
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                return {
                    'answer': "I encountered an error while summarizing the document. Please try again with a shorter text.",
                    'type': 'error',
                    'sources': []
                }
        else:
            return {
                'answer': "To summarize a legal document, please provide the document text after 'Document:' in your message.",
                'type': 'instruction',
                'sources': []
            }
    
    def _handle_legal_search_query(self, query: str, include_sources: bool) -> Dict:
        """Handle legal search and precedent queries."""
        if self.rag_system is None:
            return {
                'answer': "I don't have access to the legal database. Please ensure the RAG system is properly loaded.",
                'type': 'error',
                'sources': []
            }
        
        try:
            # Use RAG system to find relevant cases
            rag_response = self.rag_system.query(query, self.summarizer)
            
            # Format response
            answer = f"**Legal Search Results:**\n\n{rag_response['answer']}"
            
            if include_sources and rag_response['sources']:
                answer += "\n\n**Sources:**\n"
                for i, source in enumerate(rag_response['sources'][:5], 1):
                    answer += f"{i}. Case ID: {source['doc_id']} (Jurisdiction: {source['jurisdiction']}, Relevance: {source['score']:.2f})\n"
            
            return {
                'answer': answer,
                'type': 'legal_search',
                'sources': rag_response['sources'],
                'confidence': rag_response['confidence']
            }
        
        except Exception as e:
            logger.error(f"Legal search error: {e}")
            return {
                'answer': "I encountered an error while searching the legal database. Please try rephrasing your query.",
                'type': 'error',
                'sources': []
            }
    
    def _handle_case_analysis_query(self, query: str, include_sources: bool) -> Dict:
        """Handle case analysis and interpretation queries."""
        if self.rag_system is None:
            return self._handle_general_query(query, include_sources)
        
        try:
            # Get relevant context from RAG
            relevant_docs = self.rag_system.search(query, top_k=3)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find relevant legal precedents for this analysis. Could you provide more specific details?",
                    'type': 'no_results',
                    'sources': []
                }
            
            # Create analysis prompt with context
            context = "\n".join([doc['content'] for doc in relevant_docs[:2]])
            
            analysis_prompt = f"""Based on the following legal precedents, provide a detailed analysis:

Context:
{context}

Query: {query}

Please provide a comprehensive legal analysis considering:
1. Relevant legal principles
2. Applicable precedents
3. Key considerations
4. Potential implications

Analysis:"""
            
            if self.summarizer:
                analysis = self.summarizer.summarize(analysis_prompt)
            else:
                analysis = "Analysis requires access to the language model. Please ensure the summarizer is loaded."
            
            answer = f"**Legal Analysis:**\n\n{analysis}"
            
            if include_sources:
                answer += "\n\n**Based on precedents:**\n"
                for i, doc in enumerate(relevant_docs[:3], 1):
                    answer += f"{i}. {doc['metadata']['doc_id']} (Relevance: {doc['score']:.2f})\n"
            
            return {
                'answer': answer,
                'type': 'case_analysis',
                'sources': [doc['metadata'] for doc in relevant_docs],
                'confidence': sum(doc['score'] for doc in relevant_docs) / len(relevant_docs)
            }
        
        except Exception as e:
            logger.error(f"Case analysis error: {e}")
            return {
                'answer': "I encountered an error during case analysis. Please try again with a more specific query.",
                'type': 'error',
                'sources': []
            }
    
    def _handle_general_query(self, query: str, include_sources: bool) -> Dict:
        """Handle general legal queries."""
        if self.rag_system:
            # Use RAG for general queries
            try:
                rag_response = self.rag_system.query(query, self.summarizer)
                
                answer = rag_response['answer']
                if include_sources and rag_response['sources']:
                    answer += "\n\n**References:**\n"
                    for i, source in enumerate(rag_response['sources'][:3], 1):
                        answer += f"{i}. {source['doc_id']} ({source['jurisdiction']})\n"
                
                return {
                    'answer': answer,
                    'type': 'general',
                    'sources': rag_response['sources'],
                    'confidence': rag_response['confidence']
                }
            except Exception as e:
                logger.error(f"General query error: {e}")
        
        # Fallback response
        return {
            'answer': """I'm a legal AI assistant designed to help with:

1. **Document Summarization**: Provide legal documents for concise summaries
2. **Legal Research**: Search for relevant cases and precedents  
3. **Case Analysis**: Analyze legal issues and implications
4. **General Legal Questions**: Ask about legal concepts and principles

How can I assist you with your legal query today?""",
            'type': 'general',
            'sources': []
        }
    
    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.conversation_history) > self.conversation_history_limit * 2:
            self.conversation_history = self.conversation_history[-self.conversation_history_limit * 2:]
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_conversation(self, file_path: str):
        """Save conversation history to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        logger.info(f"Conversation saved to {file_path}")
    
    def load_conversation(self, file_path: str):
        """Load conversation history from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        logger.info(f"Conversation loaded from {file_path}")
    
    def get_similar_cases(self, case_description: str, top_k: int = 5) -> List[Dict]:
        """Find similar legal cases."""
        if self.rag_system is None:
            return []
        
        try:
            return self.rag_system.get_similar_cases(case_description, top_k)
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    def batch_summarize(self, documents: List[str]) -> List[Dict]:
        """Summarize multiple documents."""
        if self.summarizer is None:
            return [{'error': 'Summarizer not available'} for _ in documents]
        
        results = []
        for i, doc in enumerate(documents):
            try:
                summary = self.summarizer.summarize(doc)
                results.append({
                    'document_id': i,
                    'summary': summary,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'document_id': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return results

def main():
    """Example usage of the legal chatbot."""
    # Initialize chatbot
    chatbot = LegalChatbot()
    
    # Example conversation
    queries = [
        "What are the key principles of property tax law?",
        "Find cases related to income tax deductions",
        "Summarize this document: Appeal No. LXVI of 1949. This case deals with property tax deductions...",
        "Analyze the legal implications of municipal tax charges"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = chatbot.chat(query)
        print(f"Assistant: {response['answer'][:200]}...")
        print(f"Type: {response['type']}")

if __name__ == "__main__":
    main()
