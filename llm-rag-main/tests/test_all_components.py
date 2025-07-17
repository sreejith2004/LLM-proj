"""
Comprehensive test suite for all novel components.
"""

import pytest
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all components
from src.reasoning.cot_reasoning import ChainOfThoughtReasoner, OutcomeLabel
from src.visualization.legal_visualizer import LegalSummaryVisualizer, ViewType
from src.bias.bias_detector import BiasDetector, BiasType, SeverityLevel
from src.multilingual.multilingual_chat import MultilingualLegalChat, SupportedLanguage
from src.legal_sections.section_predictor import LegalSectionPredictor
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator, TaskType

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 COMPREHENSIVE TEST SUITE FOR ENHANCED LEGAL AI SYSTEM")
    print("=" * 70)
    
    try:
        # Initialize all components
        print("\n🔧 Initializing components...")
        cot_reasoner = ChainOfThoughtReasoner()
        visualizer = LegalSummaryVisualizer()
        bias_detector = BiasDetector()
        multilingual_chat = MultilingualLegalChat()
        section_predictor = LegalSectionPredictor()
        evaluator = ComprehensiveEvaluator()
        print("✅ All components initialized successfully")
        
        # Test data
        sample_case = """
        The appellant was charged under Section 302 IPC for the murder of the deceased on 15th January 2020. 
        The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
        The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.
        The trial court convicted the accused on 10th March 2021, but the High Court acquitted him on appeal on 5th September 2022.
        """
        
        biased_text = "The hysterical woman clearly provoked the incident, as is typical woman behavior."
        
        # Test 1: Chain-of-Thought Reasoning
        print("\n🧠 Testing Chain-of-Thought Reasoning...")
        result = cot_reasoner.analyze_case(sample_case)
        assert result is not None
        assert isinstance(result.final_verdict, OutcomeLabel)
        assert 0 <= result.overall_confidence <= 1
        print(f"   ✅ Final verdict: {result.final_verdict.value}")
        print(f"   ✅ Overall confidence: {result.overall_confidence:.2f}")
        print("   ✅ Chain-of-Thought reasoning test passed")
        
        # Test 2: Dual-View Summarizer
        print("\n📊 Testing Dual-View Summarizer...")
        dual_summary = visualizer.create_dual_view_summary(sample_case, "Test Case")
        assert dual_summary is not None
        assert dual_summary.legal_view.view_type == ViewType.LEGAL
        assert dual_summary.simplified_view.view_type == ViewType.SIMPLIFIED
        assert len(dual_summary.legal_view.content) > 0
        assert len(dual_summary.simplified_view.content) > 0
        print(f"   ✅ Legal view readability: {dual_summary.legal_view.readability_score:.2f}")
        print(f"   ✅ Simplified view readability: {dual_summary.simplified_view.readability_score:.2f}")
        print("   ✅ Dual-view summarizer test passed")
        
        # Test 3: Bias Detection
        print("\n🔍 Testing Bias Detection...")
        report = bias_detector.analyze_text(biased_text)
        assert report is not None
        assert len(report.bias_instances) > 0  # Should detect bias
        assert 0 <= report.overall_bias_score <= 1
        print(f"   ✅ Bias instances found: {len(report.bias_instances)}")
        print(f"   ✅ Overall bias score: {report.overall_bias_score:.2f}")
        print("   ✅ Bias detection test passed")
        
        # Test 4: Multilingual Chat
        print("\n🌐 Testing Multilingual Chat...")
        supported_langs = multilingual_chat.get_supported_languages()
        assert len(supported_langs) >= 10
        
        # Test English query
        response = multilingual_chat.process_query("What is bail?")
        assert response is not None
        assert len(response.english_response) > 0
        assert isinstance(response.detected_language, SupportedLanguage)
        print(f"   ✅ Supported languages: {len(supported_langs)}")
        print(f"   ✅ English query processed successfully")
        print("   ✅ Multilingual chat test passed")
        
        # Test 5: Section Predictor
        print("\n⚖️ Testing Legal Section Predictor...")
        case_facts = "The accused intentionally killed the victim with a knife after a heated argument."
        recommendation = section_predictor.predict_sections(case_facts)
        assert recommendation is not None
        assert len(recommendation.primary_sections) > 0
        assert len(recommendation.case_category) > 0
        print(f"   ✅ Primary sections predicted: {len(recommendation.primary_sections)}")
        print(f"   ✅ Case category: {recommendation.case_category}")
        print("   ✅ Section predictor test passed")
        
        # Test 6: Comprehensive Evaluator
        print("\n📈 Testing Comprehensive Evaluator...")
        predictions = ["The court ruled that the accused is guilty of murder under Section 302 IPC."]
        references = ["The Supreme Court held that the defendant committed murder as defined in Section 302 of the Indian Penal Code."]
        sources = ["The defendant killed the victim with premeditation. The Supreme Court convicted him of murder."]
        
        task_eval = evaluator.evaluate_task(TaskType.SUMMARIZATION, predictions, references, sources)
        assert task_eval is not None
        assert 0 <= task_eval.overall_score <= 1
        assert len(task_eval.results) > 0
        print(f"   ✅ Overall score: {task_eval.overall_score:.3f}")
        print(f"   ✅ Metrics evaluated: {len(task_eval.results)}")
        print("   ✅ Comprehensive evaluator test passed")
        
        # Test 7: Integration Testing
        print("\n🔗 Testing Component Integration...")
        
        # CoT + Section Predictor integration
        cot_result = cot_reasoner.analyze_case(sample_case)
        section_recommendation = section_predictor.predict_sections(sample_case)
        assert cot_result is not None
        assert section_recommendation is not None
        
        # Visualizer + Bias Detector integration
        dual_summary = visualizer.create_dual_view_summary(sample_case)
        bias_report_legal = bias_detector.analyze_text(dual_summary.legal_view.content)
        bias_report_simple = bias_detector.analyze_text(dual_summary.simplified_view.content)
        assert bias_report_simple.overall_bias_score <= bias_report_legal.overall_bias_score
        
        print("   ✅ CoT + Section Predictor integration working")
        print("   ✅ Visualizer + Bias Detector integration working")
        print("   ✅ Component integration test passed")
        
        # Test 8: Performance Benchmarks
        print("\n⚡ Testing Performance Benchmarks...")
        
        # Test CoT reasoning performance
        start_time = time.time()
        cot_reasoner.analyze_case(sample_case)
        cot_time = time.time() - start_time
        assert cot_time < 10.0  # Should complete within 10 seconds
        
        # Test dual-view summarizer performance
        start_time = time.time()
        visualizer.create_dual_view_summary(sample_case)
        viz_time = time.time() - start_time
        assert viz_time < 5.0  # Should complete within 5 seconds
        
        # Test bias detection performance
        start_time = time.time()
        bias_detector.analyze_text(biased_text)
        bias_time = time.time() - start_time
        assert bias_time < 3.0  # Should complete within 3 seconds
        
        print(f"   ✅ CoT reasoning: {cot_time:.2f}s")
        print(f"   ✅ Dual-view summarizer: {viz_time:.2f}s")
        print(f"   ✅ Bias detection: {bias_time:.2f}s")
        print("   ✅ Performance benchmarks test passed")
        
        # Final Results
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ Chain-of-Thought Reasoning: WORKING")
        print("✅ Dual-View Summarizer: WORKING")
        print("✅ Bias & Hallucination Detection: WORKING")
        print("✅ Multilingual Legal Chat: WORKING")
        print("✅ Legal Section Predictor: WORKING")
        print("✅ Comprehensive Evaluator: WORKING")
        print("✅ Component Integration: WORKING")
        print("✅ Performance Benchmarks: PASSED")
        print("\n🚀 Enhanced Legal AI System is fully operational!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\n✅ System ready for deployment!")
    else:
        print("\n❌ Please fix the issues before deployment.")
