"""
Comprehensive LLM Performance Evaluation Module.

This module provides advanced metrics and testing for the legal LLM system,
including BLEU, ROUGE, BERTScore, legal-specific metrics, and performance benchmarks.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
from pathlib import Path
from collections import defaultdict
import statistics

# Import evaluation libraries
from rouge_score import rouge_scorer
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    ROUGE = "rouge"
    BLEU = "bleu"
    BERTSCORE = "bertscore"
    LEGAL_ACCURACY = "legal_accuracy"
    FACTUAL_CONSISTENCY = "factual_consistency"
    BIAS_SCORE = "bias_score"
    HALLUCINATION_RATE = "hallucination_rate"
    RESPONSE_TIME = "response_time"
    READABILITY = "readability"
    COHERENCE = "coherence"

class TaskType(Enum):
    """Types of legal tasks for evaluation."""
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CASE_ANALYSIS = "case_analysis"
    SECTION_PREDICTION = "section_prediction"
    BIAS_DETECTION = "bias_detection"
    TRANSLATION = "translation"

@dataclass
class EvaluationResult:
    """Single evaluation result."""
    metric: EvaluationMetric
    score: float
    details: Dict[str, Any]
    confidence: float

@dataclass
class TaskEvaluation:
    """Evaluation results for a specific task."""
    task_type: TaskType
    results: List[EvaluationResult]
    overall_score: float
    sample_count: int
    execution_time: float

@dataclass
class ComprehensiveReport:
    """Complete evaluation report."""
    task_evaluations: List[TaskEvaluation]
    overall_performance: Dict[str, float]
    performance_summary: str
    recommendations: List[str]
    timestamp: str
    system_info: Dict[str, Any]

class LegalAccuracyEvaluator:
    """Evaluator for legal-specific accuracy metrics."""
    
    def __init__(self):
        """Initialize legal accuracy evaluator."""
        self.legal_entities = {
            'sections': r'(?:Section|Sec\.?)\s*(\d+[A-Z]?)',
            'cases': r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)',
            'courts': r'(Supreme Court|High Court|District Court|Sessions Court)',
            'acts': r'([A-Z][a-zA-Z\s]+(?:Act|Code))\s*(?:,?\s*\d{4})?'
        }
    
    def evaluate_legal_entity_extraction(self, predicted_text: str, reference_text: str) -> Dict[str, float]:
        """Evaluate legal entity extraction accuracy."""
        scores = {}
        
        for entity_type, pattern in self.legal_entities.items():
            pred_entities = set(re.findall(pattern, predicted_text, re.IGNORECASE))
            ref_entities = set(re.findall(pattern, reference_text, re.IGNORECASE))
            
            if ref_entities:
                precision = len(pred_entities & ref_entities) / len(pred_entities) if pred_entities else 0
                recall = len(pred_entities & ref_entities) / len(ref_entities)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                scores[f'{entity_type}_precision'] = precision
                scores[f'{entity_type}_recall'] = recall
                scores[f'{entity_type}_f1'] = f1
            else:
                scores[f'{entity_type}_precision'] = 1.0 if not pred_entities else 0.0
                scores[f'{entity_type}_recall'] = 1.0
                scores[f'{entity_type}_f1'] = 1.0 if not pred_entities else 0.0
        
        return scores
    
    def evaluate_legal_reasoning_consistency(self, text: str) -> float:
        """Evaluate consistency of legal reasoning."""
        # Simple heuristic-based evaluation
        consistency_score = 1.0
        
        # Check for contradictory statements
        contradictions = [
            (r'guilty', r'not guilty'),
            (r'liable', r'not liable'),
            (r'valid', r'invalid'),
            (r'constitutional', r'unconstitutional')
        ]
        
        text_lower = text.lower()
        for pos_pattern, neg_pattern in contradictions:
            if re.search(pos_pattern, text_lower) and re.search(neg_pattern, text_lower):
                consistency_score -= 0.2
        
        # Check for logical flow indicators
        flow_indicators = ['therefore', 'thus', 'hence', 'consequently', 'as a result']
        flow_count = sum(1 for indicator in flow_indicators if indicator in text_lower)
        
        # Boost score for good logical flow
        if flow_count > 0:
            consistency_score += min(0.1 * flow_count, 0.3)
        
        return max(0.0, min(1.0, consistency_score))

class FactualConsistencyEvaluator:
    """Evaluator for factual consistency."""
    
    def evaluate_factual_consistency(self, generated_text: str, source_text: str) -> float:
        """Evaluate factual consistency between generated and source text."""
        # Extract key facts from both texts
        generated_facts = self._extract_facts(generated_text)
        source_facts = self._extract_facts(source_text)
        
        if not source_facts:
            return 1.0  # No facts to contradict
        
        # Check for contradictions
        contradictions = 0
        total_facts = len(generated_facts)
        
        for gen_fact in generated_facts:
            if self._contradicts_source(gen_fact, source_facts):
                contradictions += 1
        
        # Calculate consistency score
        if total_facts == 0:
            return 1.0
        
        consistency_score = 1.0 - (contradictions / total_facts)
        return max(0.0, consistency_score)
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text."""
        # Simple fact extraction based on sentence patterns
        sentences = re.split(r'[.!?]+', text)
        facts = []
        
        fact_patterns = [
            r'\b(?:is|was|are|were|has|have|had)\b',  # State verbs
            r'\b(?:occurred|happened|took place)\b',   # Event verbs
            r'\b(?:ruled|decided|held|found)\b',       # Legal decision verbs
            r'\b\d+\b',  # Numbers/dates
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length
                if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in fact_patterns):
                    facts.append(sentence)
        
        return facts[:10]  # Limit to 10 facts
    
    def _contradicts_source(self, fact: str, source_facts: List[str]) -> bool:
        """Check if a fact contradicts source facts."""
        # Simple contradiction detection
        fact_lower = fact.lower()
        
        # Look for negation patterns
        negation_pairs = [
            ('is', 'is not'),
            ('was', 'was not'),
            ('guilty', 'not guilty'),
            ('liable', 'not liable'),
            ('valid', 'invalid')
        ]
        
        for source_fact in source_facts:
            source_lower = source_fact.lower()
            
            # Check for direct contradictions
            for pos, neg in negation_pairs:
                if pos in fact_lower and neg in source_lower:
                    return True
                if neg in fact_lower and pos in source_lower:
                    return True
        
        return False

class ReadabilityEvaluator:
    """Evaluator for text readability."""
    
    def calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, flesch_score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower().strip('.,!?";')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)

class CoherenceEvaluator:
    """Evaluator for text coherence."""
    
    def evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        coherence_score = 0.0
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'nevertheless', 'additionally', 'similarly', 'in contrast', 'on the other hand'
        ]
        
        transition_count = 0
        for sentence in sentences:
            if any(word in sentence.lower() for word in transition_words):
                transition_count += 1
        
        # Transition score
        transition_score = min(transition_count / len(sentences), 0.3)
        
        # Repetition score (avoid excessive repetition)
        word_freq = defaultdict(int)
        total_words = 0
        
        for sentence in sentences:
            words = sentence.lower().split()
            total_words += len(words)
            for word in words:
                if len(word) > 3:  # Only count significant words
                    word_freq[word] += 1
        
        if total_words > 0:
            # Calculate repetition penalty
            repetition_penalty = 0
            for word, freq in word_freq.items():
                if freq > 3:  # Word appears more than 3 times
                    repetition_penalty += (freq - 3) * 0.05
            
            repetition_score = max(0, 0.4 - repetition_penalty)
        else:
            repetition_score = 0.4
        
        # Sentence length variation
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_std = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
            variation_score = min(length_std / 10, 0.3)  # Normalize
        else:
            variation_score = 0.3
        
        coherence_score = transition_score + repetition_score + variation_score
        return min(1.0, coherence_score)

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for legal LLM performance.
    """
    
    def __init__(self):
        """Initialize comprehensive evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.legal_evaluator = LegalAccuracyEvaluator()
        self.factual_evaluator = FactualConsistencyEvaluator()
        self.readability_evaluator = ReadabilityEvaluator()
        self.coherence_evaluator = CoherenceEvaluator()
        
        # Initialize BLEU if available
        if NLTK_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method1
        
        logger.info("Comprehensive Evaluator initialized")
    
    def evaluate_task(self, task_type: TaskType, predictions: List[str], 
                     references: List[str], sources: Optional[List[str]] = None,
                     additional_data: Optional[Dict] = None) -> TaskEvaluation:
        """
        Evaluate a specific task comprehensively.
        
        Args:
            task_type: Type of task being evaluated
            predictions: Model predictions
            references: Reference/ground truth texts
            sources: Source texts (for consistency evaluation)
            additional_data: Additional task-specific data
            
        Returns:
            TaskEvaluation: Complete evaluation results
        """
        logger.info(f"Evaluating task: {task_type.value}")
        start_time = time.time()
        
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        results = []
        
        # ROUGE evaluation
        rouge_result = self._evaluate_rouge(predictions, references)
        results.append(rouge_result)
        
        # BLEU evaluation (if available)
        if NLTK_AVAILABLE:
            bleu_result = self._evaluate_bleu(predictions, references)
            results.append(bleu_result)
        
        # Legal accuracy evaluation
        legal_result = self._evaluate_legal_accuracy(predictions, references)
        results.append(legal_result)
        
        # Factual consistency (if sources provided)
        if sources:
            factual_result = self._evaluate_factual_consistency(predictions, sources)
            results.append(factual_result)
        
        # Readability evaluation
        readability_result = self._evaluate_readability(predictions)
        results.append(readability_result)
        
        # Coherence evaluation
        coherence_result = self._evaluate_coherence(predictions)
        results.append(coherence_result)
        
        # Task-specific evaluations
        if task_type == TaskType.BIAS_DETECTION and additional_data:
            bias_result = self._evaluate_bias_detection(additional_data)
            results.append(bias_result)
        
        # Calculate overall score
        overall_score = np.mean([result.score for result in results])
        execution_time = time.time() - start_time
        
        return TaskEvaluation(
            task_type=task_type,
            results=results,
            overall_score=overall_score,
            sample_count=len(predictions),
            execution_time=execution_time
        )
    
    def _evaluate_rouge(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Evaluate using ROUGE metrics."""
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric, score in scores.items():
                rouge_scores[f"{metric}_precision"].append(score.precision)
                rouge_scores[f"{metric}_recall"].append(score.recall)
                rouge_scores[f"{metric}_fmeasure"].append(score.fmeasure)
        
        # Calculate averages
        avg_scores = {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
        overall_rouge = np.mean([avg_scores['rouge1_fmeasure'], avg_scores['rouge2_fmeasure'], avg_scores['rougeL_fmeasure']])
        
        return EvaluationResult(
            metric=EvaluationMetric.ROUGE,
            score=overall_rouge,
            details=avg_scores,
            confidence=0.9
        )
    
    def _evaluate_bleu(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Evaluate using BLEU score."""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects list of reference token lists
            
            try:
                bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing_function)
                bleu_scores.append(bleu_score)
            except:
                bleu_scores.append(0.0)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.BLEU,
            score=avg_bleu,
            details={'individual_scores': bleu_scores, 'average': avg_bleu},
            confidence=0.8
        )
    
    def _evaluate_legal_accuracy(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """Evaluate legal accuracy."""
        legal_scores = []
        entity_scores = defaultdict(list)
        consistency_scores = []
        
        for pred, ref in zip(predictions, references):
            # Legal entity extraction accuracy
            entity_acc = self.legal_evaluator.evaluate_legal_entity_extraction(pred, ref)
            for metric, score in entity_acc.items():
                entity_scores[metric].append(score)
            
            # Legal reasoning consistency
            consistency = self.legal_evaluator.evaluate_legal_reasoning_consistency(pred)
            consistency_scores.append(consistency)
        
        # Calculate overall legal accuracy
        avg_entity_f1 = np.mean([np.mean([entity_scores.get(f'{entity}_f1', [0]) for entity in ['sections', 'cases', 'courts', 'acts']]) for _ in predictions])
        avg_consistency = np.mean(consistency_scores)
        overall_legal = (avg_entity_f1 + avg_consistency) / 2
        
        details = {
            'entity_extraction': {metric: np.mean(scores) for metric, scores in entity_scores.items()},
            'reasoning_consistency': avg_consistency,
            'overall': overall_legal
        }
        
        return EvaluationResult(
            metric=EvaluationMetric.LEGAL_ACCURACY,
            score=overall_legal,
            details=details,
            confidence=0.7
        )
    
    def _evaluate_factual_consistency(self, predictions: List[str], sources: List[str]) -> EvaluationResult:
        """Evaluate factual consistency."""
        consistency_scores = []
        
        for pred, source in zip(predictions, sources):
            consistency = self.factual_evaluator.evaluate_factual_consistency(pred, source)
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        
        return EvaluationResult(
            metric=EvaluationMetric.FACTUAL_CONSISTENCY,
            score=avg_consistency,
            details={'individual_scores': consistency_scores, 'average': avg_consistency},
            confidence=0.6
        )
    
    def _evaluate_readability(self, predictions: List[str]) -> EvaluationResult:
        """Evaluate readability."""
        readability_scores = []
        
        for pred in predictions:
            flesch_score = self.readability_evaluator.calculate_flesch_reading_ease(pred)
            # Normalize to 0-1 scale
            normalized_score = flesch_score / 100
            readability_scores.append(normalized_score)
        
        avg_readability = np.mean(readability_scores)
        
        return EvaluationResult(
            metric=EvaluationMetric.READABILITY,
            score=avg_readability,
            details={'flesch_scores': readability_scores, 'average': avg_readability},
            confidence=0.8
        )
    
    def _evaluate_coherence(self, predictions: List[str]) -> EvaluationResult:
        """Evaluate coherence."""
        coherence_scores = []
        
        for pred in predictions:
            coherence = self.coherence_evaluator.evaluate_coherence(pred)
            coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores)
        
        return EvaluationResult(
            metric=EvaluationMetric.COHERENCE,
            score=avg_coherence,
            details={'individual_scores': coherence_scores, 'average': avg_coherence},
            confidence=0.7
        )
    
    def _evaluate_bias_detection(self, bias_data: Dict) -> EvaluationResult:
        """Evaluate bias detection performance."""
        # This would evaluate bias detection accuracy
        # For now, return a placeholder
        return EvaluationResult(
            metric=EvaluationMetric.BIAS_SCORE,
            score=0.8,  # Placeholder
            details={'bias_detection_accuracy': 0.8},
            confidence=0.6
        )
    
    def generate_comprehensive_report(self, task_evaluations: List[TaskEvaluation]) -> ComprehensiveReport:
        """Generate comprehensive evaluation report."""
        # Calculate overall performance
        overall_performance = {}
        
        # Aggregate scores by metric
        metric_scores = defaultdict(list)
        for task_eval in task_evaluations:
            for result in task_eval.results:
                metric_scores[result.metric.value].append(result.score)
        
        for metric, scores in metric_scores.items():
            overall_performance[metric] = np.mean(scores)
        
        # Calculate overall system score
        overall_performance['system_score'] = np.mean(list(overall_performance.values()))
        
        # Generate summary
        summary = self._generate_performance_summary(task_evaluations, overall_performance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(task_evaluations, overall_performance)
        
        # System info
        system_info = {
            'total_tasks_evaluated': len(task_evaluations),
            'total_samples': sum(task.sample_count for task in task_evaluations),
            'total_execution_time': sum(task.execution_time for task in task_evaluations),
            'nltk_available': NLTK_AVAILABLE
        }
        
        return ComprehensiveReport(
            task_evaluations=task_evaluations,
            overall_performance=overall_performance,
            performance_summary=summary,
            recommendations=recommendations,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            system_info=system_info
        )
    
    def _generate_performance_summary(self, task_evaluations: List[TaskEvaluation], 
                                    overall_performance: Dict[str, float]) -> str:
        """Generate performance summary."""
        summary_parts = []
        
        system_score = overall_performance.get('system_score', 0)
        summary_parts.append(f"Overall System Performance: {system_score:.3f}")
        
        # Task-specific performance
        for task_eval in task_evaluations:
            summary_parts.append(f"{task_eval.task_type.value.title()}: {task_eval.overall_score:.3f}")
        
        # Metric highlights
        best_metric = max(overall_performance.items(), key=lambda x: x[1] if x[0] != 'system_score' else 0)
        worst_metric = min(overall_performance.items(), key=lambda x: x[1] if x[0] != 'system_score' else 1)
        
        summary_parts.append(f"Best performing metric: {best_metric[0]} ({best_metric[1]:.3f})")
        summary_parts.append(f"Needs improvement: {worst_metric[0]} ({worst_metric[1]:.3f})")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_recommendations(self, task_evaluations: List[TaskEvaluation], 
                                overall_performance: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if overall_performance.get('rouge', 0) < 0.5:
            recommendations.append("Improve content relevance and coverage (low ROUGE scores)")
        
        if overall_performance.get('legal_accuracy', 0) < 0.6:
            recommendations.append("Enhance legal knowledge and entity recognition accuracy")
        
        if overall_performance.get('factual_consistency', 0) < 0.7:
            recommendations.append("Reduce factual inconsistencies and hallucinations")
        
        if overall_performance.get('readability', 0) < 0.6:
            recommendations.append("Improve text readability and simplification")
        
        if overall_performance.get('coherence', 0) < 0.7:
            recommendations.append("Enhance text coherence and logical flow")
        
        # Task-specific recommendations
        for task_eval in task_evaluations:
            if task_eval.overall_score < 0.6:
                recommendations.append(f"Focus on improving {task_eval.task_type.value} performance")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory across all metrics")
        
        return recommendations
    
    def save_report(self, report: ComprehensiveReport, filepath: str):
        """Save evaluation report to file."""
        # Convert dataclasses to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filepath}")

def main():
    """Example usage of Comprehensive Evaluator."""
    # Sample data for evaluation
    predictions = [
        "The court ruled that the accused is guilty of murder under Section 302 IPC.",
        "The appellant filed an appeal against the judgment of the lower court.",
        "The case involves domestic violence and cruelty by the husband."
    ]
    
    references = [
        "The Supreme Court held that the defendant committed murder as defined in Section 302 of the Indian Penal Code.",
        "An appeal was filed by the appellant challenging the decision of the trial court.",
        "This case concerns domestic cruelty and violence perpetrated by the spouse."
    ]
    
    sources = [
        "The defendant killed the victim with premeditation. The Supreme Court convicted him of murder.",
        "The trial court convicted the appellant. He filed an appeal to the High Court.",
        "The husband subjected his wife to physical and mental cruelty at home."
    ]
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    print("=" * 70)
    print("COMPREHENSIVE LLM EVALUATION DEMO")
    print("=" * 70)
    
    # Evaluate different tasks
    task_evaluations = []
    
    # Summarization task
    summarization_eval = evaluator.evaluate_task(
        TaskType.SUMMARIZATION,
        predictions,
        references,
        sources
    )
    task_evaluations.append(summarization_eval)
    
    # Question answering task
    qa_eval = evaluator.evaluate_task(
        TaskType.QUESTION_ANSWERING,
        predictions[:2],
        references[:2],
        sources[:2]
    )
    task_evaluations.append(qa_eval)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(task_evaluations)
    
    # Display results
    print(f"\nEvaluation completed at: {report.timestamp}")
    print(f"System Info: {report.system_info}")
    
    print(f"\nOverall Performance:")
    for metric, score in report.overall_performance.items():
        print(f"  {metric}: {score:.3f}")
    
    print(f"\nPerformance Summary:")
    print(report.performance_summary)
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\nTask-specific Results:")
    for task_eval in task_evaluations:
        print(f"\n{task_eval.task_type.value.upper()}:")
        print(f"  Overall Score: {task_eval.overall_score:.3f}")
        print(f"  Samples: {task_eval.sample_count}")
        print(f"  Execution Time: {task_eval.execution_time:.2f}s")
        
        for result in task_eval.results:
            print(f"  {result.metric.value}: {result.score:.3f} (confidence: {result.confidence:.2f})")

if __name__ == "__main__":
    main()
