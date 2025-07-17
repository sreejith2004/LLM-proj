"""
Evaluation utilities for legal document summarization and RAG system.
"""

import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class LegalEvaluator:
    """
    Evaluation utilities for legal AI system.
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_summarization(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate summarization quality using ROUGE scores.
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            
            for metric, score in scores.items():
                rouge_scores[f"{metric}_precision"].append(score.precision)
                rouge_scores[f"{metric}_recall"].append(score.recall)
                rouge_scores[f"{metric}_fmeasure"].append(score.fmeasure)
        
        # Calculate averages
        avg_scores = {}
        for metric, scores in rouge_scores.items():
            avg_scores[metric] = np.mean(scores)
        
        return avg_scores
    
    def evaluate_rag_retrieval(self, queries: List[str], 
                              retrieved_docs: List[List[Dict]], 
                              relevant_docs: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate RAG retrieval quality.
        """
        if len(queries) != len(retrieved_docs) != len(relevant_docs):
            raise ValueError("All input lists must have the same length")
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []  # Mean Reciprocal Rank
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_ids = [doc['metadata']['doc_id'] for doc in retrieved]
            relevant_set = set(relevant)
            retrieved_set = set(retrieved_ids)
            
            # Precision and Recall
            if retrieved_set:
                precision = len(relevant_set & retrieved_set) / len(retrieved_set)
                precision_scores.append(precision)
            
            if relevant_set:
                recall = len(relevant_set & retrieved_set) / len(relevant_set)
                recall_scores.append(recall)
            
            # F1 Score
            if precision_scores and recall_scores:
                p, r = precision_scores[-1], recall_scores[-1]
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                f1_scores.append(f1)
            
            # Mean Reciprocal Rank
            rr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    rr = 1 / (i + 1)
                    break
            mrr_scores.append(rr)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'f1': np.mean(f1_scores) if f1_scores else 0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0
        }
    
    def evaluate_legal_qa(self, questions: List[str], 
                         predictions: List[str], 
                         references: List[str]) -> Dict[str, float]:
        """
        Evaluate legal question answering.
        """
        # Use ROUGE for semantic similarity
        rouge_scores = self.evaluate_summarization(predictions, references)
        
        # Additional legal-specific metrics could be added here
        # e.g., legal entity extraction accuracy, citation accuracy, etc.
        
        return rouge_scores
    
    def calculate_legal_metrics(self, predictions: List[str], 
                               references: List[str]) -> Dict[str, float]:
        """
        Calculate legal-specific evaluation metrics.
        """
        legal_terms = [
            'court', 'judge', 'appellant', 'respondent', 'petitioner',
            'section', 'act', 'law', 'case', 'judgment', 'appeal',
            'liability', 'deduction', 'tax', 'property', 'income'
        ]
        
        legal_term_precision = []
        legal_term_recall = []
        
        for pred, ref in zip(predictions, references):
            pred_terms = set(term for term in legal_terms if term.lower() in pred.lower())
            ref_terms = set(term for term in legal_terms if term.lower() in ref.lower())
            
            if pred_terms:
                precision = len(pred_terms & ref_terms) / len(pred_terms)
                legal_term_precision.append(precision)
            
            if ref_terms:
                recall = len(pred_terms & ref_terms) / len(ref_terms)
                legal_term_recall.append(recall)
        
        return {
            'legal_term_precision': np.mean(legal_term_precision) if legal_term_precision else 0,
            'legal_term_recall': np.mean(legal_term_recall) if legal_term_recall else 0
        }

def main():
    """Example usage of the evaluator."""
    evaluator = LegalEvaluator()
    
    # Example data
    predictions = [
        "The court ruled in favor of the appellant regarding property tax deductions.",
        "The case involves income tax liability and municipal charges."
    ]
    
    references = [
        "The Supreme Court decided that property tax deductions are allowable under section 9.",
        "This appeal concerns income tax assessment and municipal tax liability."
    ]
    
    # Evaluate summarization
    rouge_scores = evaluator.evaluate_summarization(predictions, references)
    print("ROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Evaluate legal metrics
    legal_scores = evaluator.calculate_legal_metrics(predictions, references)
    print("\nLegal Metrics:")
    for metric, score in legal_scores.items():
        print(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    main()
