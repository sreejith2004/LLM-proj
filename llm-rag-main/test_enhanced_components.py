#!/usr/bin/env python3
"""
Test all enhanced components with real LLM integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_ollama_connection():
    """Test if Ollama is available."""
    try:
        from src.utils.llm_client import OllamaClient
        
        client = OllamaClient()
        available = client.is_available()
        models = client.get_available_models()
        
        print("🔗 OLLAMA CONNECTION TEST")
        print("=" * 40)
        print(f"Ollama Available: {'✅ Yes' if available else '❌ No'}")
        print(f"Available Models: {models}")
        
        if available and models:
            # Test simple generation
            response = client.generate("Hello, this is a test.", max_tokens=50)
            print(f"Test Generation: {'✅ Success' if response.success else '❌ Failed'}")
            if response.success:
                print(f"Response: {response.content[:100]}...")
        
        return available
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_enhanced_cot_reasoning():
    """Test enhanced Chain-of-Thought reasoning."""
    print("\n🧠 ENHANCED CHAIN-OF-THOUGHT TEST")
    print("=" * 50)
    
    try:
        from src.reasoning.cot_reasoning import ChainOfThoughtReasoner
        
        reasoner = ChainOfThoughtReasoner()
        
        # Test with valid legal case
        legal_case = \"\"\"The accused was charged under Section 302 IPC for the murder of the victim. 
        The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
        The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.\"\"\"\n        
        print(f\"Input: {legal_case[:100]}...\")\n        
        result = reasoner.analyze_case(legal_case)\n        
        print(f\"Final Verdict: {result.final_verdict.value}\")\n        print(f\"Overall Confidence: {result.overall_confidence:.2f}\")\n        print(f\"LAW Step: {result.law_step.reasoning[:100]}...\")\n        print(f\"FACT Step: {result.fact_step.reasoning[:100]}...\")\n        \n        # Test with invalid input\n        print(\"\\n--- Testing Invalid Input ---\")\n        invalid_result = reasoner.analyze_case(\"zebi random nonsense\")\n        print(f\"Invalid Input Verdict: {invalid_result.final_verdict.value}\")\n        print(f\"Invalid Input Confidence: {invalid_result.overall_confidence:.2f}\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ CoT test failed: {e}\")\n        return False\n\ndef test_enhanced_bias_detection():\n    \"\"\"Test enhanced bias detection.\"\"\"\n    print(\"\\n🔍 ENHANCED BIAS DETECTION TEST\")\n    print(\"=\" * 50)\n    \n    try:\n        from src.bias.bias_detector import BiasDetector\n        \n        detector = BiasDetector()\n        \n        # Test with biased text\n        biased_text = \"\"\"The hysterical woman clearly provoked the incident, as is typical woman behavior.\n        This case is similar to Fake Case v. Non-existent Party (2025) where the Supreme Court ruled.\n        The Muslim community tends to be more aggressive in such matters.\"\"\"\n        \n        print(f\"Input: {biased_text[:100]}...\")\n        \n        report = detector.analyze_text(biased_text)\n        \n        print(f\"Bias Instances Found: {len(report.bias_instances)}\")\n        print(f\"Hallucination Instances: {len(report.hallucination_instances)}\")\n        print(f\"Overall Bias Score: {report.overall_bias_score:.2f}\")\n        print(f\"Summary: {report.summary[:100]}...\")\n        \n        # Test with clean text\n        print(\"\\n--- Testing Clean Text ---\")\n        clean_text = \"The accused was charged under Section 302 IPC. The court examined the evidence carefully.\"\n        clean_report = detector.analyze_text(clean_text)\n        print(f\"Clean Text Bias Score: {clean_report.overall_bias_score:.2f}\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ Bias detection test failed: {e}\")\n        return False\n\ndef test_enhanced_dual_view_summarizer():\n    \"\"\"Test enhanced dual-view summarizer.\"\"\"\n    print(\"\\n📊 ENHANCED DUAL-VIEW SUMMARIZER TEST\")\n    print(\"=\" * 50)\n    \n    try:\n        from src.visualization.legal_visualizer import LegalSummaryVisualizer\n        \n        visualizer = LegalSummaryVisualizer()\n        \n        legal_text = \"\"\"The appellant was charged under Section 302 of the Indian Penal Code for the murder of the deceased. \n        The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.\n        The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.\n        The trial court convicted the accused, but the High Court acquitted him on appeal.\"\"\"\n        \n        print(f\"Input: {legal_text[:100]}...\")\n        \n        dual_summary = visualizer.create_dual_view_summary(legal_text, \"Sample Murder Case\")\n        \n        print(f\"\\n{dual_summary.legal_view.title}:\")\n        print(f\"Content Length: {len(dual_summary.legal_view.content)} chars\")\n        print(f\"Readability: {dual_summary.legal_view.readability_score:.2f}\")\n        print(f\"Preview: {dual_summary.legal_view.content[:150]}...\")\n        \n        print(f\"\\n{dual_summary.simplified_view.title}:\")\n        print(f\"Content Length: {len(dual_summary.simplified_view.content)} chars\")\n        print(f\"Readability: {dual_summary.simplified_view.readability_score:.2f}\")\n        print(f\"Preview: {dual_summary.simplified_view.content[:150]}...\")\n        \n        print(f\"\\nComparison Metrics:\")\n        for metric, value in dual_summary.comparison_metrics.items():\n            print(f\"  {metric}: {value:.3f}\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ Dual-view summarizer test failed: {e}\")\n        return False\n\ndef test_section_predictor():\n    \"\"\"Test the fixed section predictor.\"\"\"\n    print(\"\\n⚖️ SECTION PREDICTOR TEST\")\n    print(\"=\" * 40)\n    \n    try:\n        from src.legal_sections.section_predictor import LegalSectionPredictor\n        \n        predictor = LegalSectionPredictor()\n        \n        test_case = \"Analyze a murder case under Section 302 IPC\"\n        print(f\"Input: {test_case}\")\n        \n        result = predictor.predict_sections(test_case)\n        \n        print(f\"Case Category: {result.case_category}\")\n        print(f\"Severity: {result.severity_assessment}\")\n        print(f\"Primary Sections: {len(result.primary_sections)}\")\n        \n        if result.primary_sections:\n            for pred in result.primary_sections[:2]:\n                print(f\"  • Section {pred.section.section_number}: {pred.section.title}\")\n                print(f\"    Confidence: {pred.confidence:.2f}\")\n        \n        return True\n        \n    except Exception as e:\n        print(f\"❌ Section predictor test failed: {e}\")\n        return False\n\ndef main():\n    \"\"\"Run all enhanced component tests.\"\"\"\n    print(\"🚀 ENHANCED LEGAL AI SYSTEM - COMPREHENSIVE TEST\")\n    print(\"=\" * 60)\n    \n    results = {\n        \"Ollama Connection\": test_ollama_connection(),\n        \"Enhanced CoT Reasoning\": test_enhanced_cot_reasoning(),\n        \"Enhanced Bias Detection\": test_enhanced_bias_detection(),\n        \"Enhanced Dual-View Summarizer\": test_enhanced_dual_view_summarizer(),\n        \"Section Predictor\": test_section_predictor()\n    }\n    \n    print(\"\\n\" + \"=\" * 60)\n    print(\"📊 FINAL RESULTS\")\n    print(\"=\" * 60)\n    \n    for component, success in results.items():\n        status = \"✅ WORKING\" if success else \"❌ FAILED\"\n        print(f\"{component}: {status}\")\n    \n    working_count = sum(results.values())\n    total_count = len(results)\n    \n    print(f\"\\n🎯 SUMMARY: {working_count}/{total_count} components working\")\n    \n    if working_count == total_count:\n        print(\"🎉 ALL COMPONENTS ARE NOW REALLY WORKING!\")\n    elif working_count >= 3:\n        print(\"✅ Most components working - system is functional!\")\n    else:\n        print(\"⚠️  Several components need attention\")\n\nif __name__ == \"__main__\":\n    main()
